"""
InkFlow AI Enhancement Layer
Prompt optimization, multi-step reasoning, context building, response caching
"""

import httpx
import json
from datetime import datetime
from typing import Dict, List, Optional
from config import (
    GEMINI_API_KEYS, GEMINI_MODEL_FLASH, GEMINI_MODEL_PRO, 
    GEMINI_API_URL, BUSINESS_CONTEXT
)
from db.database import local_db

class AIEnhancer:
    """Enhances AI interactions with caching, context, and optimization"""
    
    def __init__(self):
        self.current_key_index = 0
        self.call_history = []
    
    # ========== GEMINI API CALL ==========
    
    async def call_gemini(self, prompt: str, model: str = None, max_tokens: int = 800) -> Optional[str]:
        """Call Gemini API with key rotation"""
        model = model or GEMINI_MODEL_FLASH
        
        # Check cache first
        cached = local_db.get_cached_response(prompt)
        if cached:
            return cached['response']
        
        # Check semantic cache
        similar = local_db.search_similar_cache(prompt)
        if similar:
            return f"{similar['response']}\n\n💾 (Retrieved from similar cached response)"
        
        # Try each key
        for attempt in range(len(GEMINI_API_KEYS)):
            key_index = (self.current_key_index + attempt) % len(GEMINI_API_KEYS)
            api_key = GEMINI_API_KEYS[key_index]
            
            url = GEMINI_API_URL.format(model=model, key=api_key)
            
            request_body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": max_tokens
                }
            }
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json=request_body)
                    
                    if response.status_code == 200:
                        data = response.json()
                        reply = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        
                        if reply:
                            # Cache the response
                            local_db.cache_gemini_response(prompt, reply, model, max_tokens)
                            self.current_key_index = key_index
                            self.call_history.append({
                                "timestamp": datetime.now().isoformat(),
                                "model": model,
                                "tokens": max_tokens,
                                "key_index": key_index
                            })
                            return reply
                    
                    elif response.status_code == 429:
                        # Rate limited, try next key
                        continue
                    
            except Exception as e:
                print(f"Gemini call error: {e}")
                continue
        
        return None
    
    # ========== PROMPT OPTIMIZATION ==========
    
    def build_optimized_prompt(self, user_query: str, context: Dict = None, mode: str = "general") -> str:
        """Build an optimized prompt with all relevant context"""
        
        # Base system prompt
        system_prompt = f"""You are InkFlow Assistant, an AI business advisor for {BUSINESS_CONTEXT['company']}.
Founder: {BUSINESS_CONTEXT['founder']}
Product: {BUSINESS_CONTEXT['product']}
Stage: {BUSINESS_CONTEXT['current_stage']}
Pricing: ${BUSINESS_CONTEXT['pricing']['standard']}/${BUSINESS_CONTEXT['pricing']['pro']}/${BUSINESS_CONTEXT['pricing']['premium']}/month
Competitors: {', '.join(BUSINESS_CONTEXT['competitors'])}
Vision: {BUSINESS_CONTEXT['vision']}

"""
        
        # Add mode-specific instructions
        mode_instructions = {
            "analytics": "Focus on data, metrics, and quantitative analysis. Be precise with numbers.",
            "outreach": "Focus on messaging, lead generation, and communication strategy.",
            "strategy": "Focus on business decisions, competitive positioning, and growth tactics.",
            "general": "Be conversational and helpful across all topics."
        }
        system_prompt += f"Mode: {mode}. {mode_instructions.get(mode, mode_instructions['general'])}\n\n"
        
        # Add dashboard context if available
        if context and context.get('dashboard'):
            d = context['dashboard']
            system_prompt += f"""Current Dashboard Data:
- {d.get('totalArtists', 0)} artists ({d.get('activeArtists', 0)} active)
- MRR: ${d.get('mrr', 0)}
- {d.get('totalBookings', 0)} total bookings ({d.get('bookings7d', 0)} this week)
- {d.get('dmsSent', 0)} DMs sent ({d.get('replyRate', 0)}% reply rate)
- Health: {d.get('healthScore', 0)}%

"""
        
        # Add ML insights if available
        if context and context.get('ml_insights'):
            system_prompt += f"""ML Insights:
- {context['ml_insights']}

"""
        
        # Add conversation history summary
        if context and context.get('recent_conversations'):
            system_prompt += f"Recent conversation topics: {', '.join(context['recent_conversations'][:5])}\n\n"
        
        # Final prompt
        full_prompt = f"{system_prompt}User Question: {user_query}\n\nProvide a thorough, actionable response. Complete all sentences. Be specific with numbers when available."
        
        return full_prompt
    
    # ========== MULTI-STEP REASONING ==========
    
    async def multi_step_analysis(self, question: str, dashboard_data: Dict) -> Dict:
        """Perform multi-step reasoning for complex questions"""
        
        results = {
            "question": question,
            "steps": [],
            "final_answer": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: Understand the question
        step1_prompt = f"Analyze this business question about a SaaS startup: '{question}'. What specific data points and analysis would be needed to answer it thoroughly? Keep it brief."
        step1_response = await self.call_gemini(step1_prompt, max_tokens=200)
        results['steps'].append({"step": "question_analysis", "result": step1_response})
        
        # Step 2: Analyze the data
        data_summary = json.dumps(dashboard_data, indent=2)[:2000]
        step2_prompt = f"Here is the current dashboard data:\n{data_summary}\n\nBased on the question '{question}', what are the key findings from this data? Be specific with numbers."
        step2_response = await self.call_gemini(step2_prompt, max_tokens=300)
        results['steps'].append({"step": "data_analysis", "result": step2_response})
        
        # Step 3: Generate recommendations
        step3_prompt = f"Question: {question}\n\nData findings: {step2_response}\n\nBased on this analysis, what are 3-5 specific, actionable recommendations for the founder? Prioritize by impact."
        step3_response = await self.call_gemini(step3_prompt, max_tokens=400)
        results['steps'].append({"step": "recommendations", "result": step3_response})
        
        # Step 4: Compile final answer
        final_prompt = f"""Question: {question}

Analysis: {step1_response}

Data Findings: {step2_response}

Recommendations: {step3_response}

Combine all of this into one comprehensive, well-structured answer. Include:
1. Brief analysis of the situation
2. Key data points
3. Actionable recommendations
4. Next steps

Be thorough but concise."""
        
        final_answer = await self.call_gemini(final_prompt, max_tokens=800)
        results['final_answer'] = final_answer
        
        return results
    
    # ========== CONTEXT BUILDER ==========
    
    def build_context(self, firebase_client, local_db) -> Dict:
        """Build comprehensive context for AI prompts"""
        context = {}
        
        try:
            # Get dashboard snapshot
            from db.firebase_client import firebase_client as fc
            snapshot = fc.take_snapshot()
            context['dashboard'] = snapshot
        except:
            context['dashboard'] = {}
        
        try:
            # Get ML insights
            if local_db:
                recent_predictions = local_db.get_predictions_for_training()
                if recent_predictions:
                    context['ml_insights'] = f"{len(recent_predictions)} predictions tracked"
        except:
            pass
        
        try:
            # Get recent conversation topics
            if local_db:
                # This would query conversations table
                context['recent_conversations'] = ["business metrics", "outreach strategy"]
        except:
            pass
        
        return context
    
    # ========== FALLBACK CHAIN ==========
    
    async def get_best_response(self, question: str, context: Dict = None) -> Dict:
        """Try multiple AI approaches, return the best result"""
        
        # Try 1: Cached response
        cached = local_db.get_cached_response(question)
        if cached:
            return {"response": cached['response'], "source": "cache", "model": cached.get('model_used', 'unknown')}
        
        # Try 2: Gemini Flash (fast, cheap)
        flash_prompt = self.build_optimized_prompt(question, context, "general")
        flash_response = await self.call_gemini(flash_prompt, model=GEMINI_MODEL_FLASH, max_tokens=600)
        if flash_response:
            return {"response": flash_response, "source": "gemini_flash", "model": GEMINI_MODEL_FLASH}
        
        # Try 3: Gemini Pro (smarter, more expensive)
        pro_prompt = self.build_optimized_prompt(question, context, "general")
        pro_response = await self.call_gemini(pro_prompt, model=GEMINI_MODEL_PRO, max_tokens=800)
        if pro_response:
            return {"response": pro_response, "source": "gemini_pro", "model": GEMINI_MODEL_PRO}
        
        # Try 4: Multi-step reasoning for complex questions
        if len(question.split()) > 8:
            try:
                dashboard_data = context.get('dashboard', {}) if context else {}
                multi_step_result = await self.multi_step_analysis(question, dashboard_data)
                if multi_step_result.get('final_answer'):
                    return {"response": multi_step_result['final_answer'], "source": "multi_step", "model": GEMINI_MODEL_FLASH}
            except:
                pass
        
        # Fallback: Return None (caller should use offline brain)
        return {"response": None, "source": "failed", "model": None}


# Singleton instance
ai_enhancer = AIEnhancer()
