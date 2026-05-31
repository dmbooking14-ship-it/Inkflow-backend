import firebase_admin
from firebase_admin import credentials, firestore
from config import (
    FIREBASE_PROJECT_ID,
    FIREBASE_KEY_PATH,
    COLLECTION_ARTISTS,
    COLLECTION_BOOKINGS,
    COLLECTION_OUTREACH_LOGS,
    COLLECTION_PENDING_PAYMENTS,
    COLLECTION_WHATSAPP_STATS,
    COLLECTION_SIDE_HUSTLE,
    COLLECTION_HONEYPOT_LOGS,
    COLLECTION_COMPETITION_WINNERS,
    COLLECTION_DAILY_SNAPSHOTS,
)
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List


class FirebaseClient:
    """Singleton Firebase client for all Firestore operations."""

    _instance: Optional["FirebaseClient"] = None

    def __new__(cls) -> "FirebaseClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.db = self._connect()

    def _connect(self):
        """Connect to Firebase Firestore."""
        try:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(
                cred, {"projectId": FIREBASE_PROJECT_ID}
            )
            db = firestore.client()
            print("✅ Firebase connected")
            return db
        except ValueError:
            # Already initialized (e.g., after hot reload)
            return firestore.client()
        except Exception as e:
            print(f"❌ Firebase connection failed: {e}")
            raise

    def _collection_to_list(self, collection_name: str) -> List[Dict[str, Any]]:
        """Fetch all documents from a collection and return as list of dicts."""
        docs = self.db.collection(collection_name).stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            results.append(data)
        return results

    # ========== ARTISTS ==========
    def get_all_artists(self) -> List[Dict[str, Any]]:
        return self._collection_to_list(COLLECTION_ARTISTS)

    # ========== BOOKINGS ==========
    def get_all_bookings(self) -> List[Dict[str, Any]]:
        return self._collection_to_list(COLLECTION_BOOKINGS)

    # ========== OUTREACH LOGS ==========
    def get_outreach_logs(self, limit: int = 200) -> List[Dict[str, Any]]:
        docs = (
            self.db.collection(COLLECTION_OUTREACH_LOGS)
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            results.append(data)
        return results

    # ========== PENDING PAYMENTS ==========
    def get_pending_payments(self) -> List[Dict[str, Any]]:
        docs = (
            self.db.collection(COLLECTION_PENDING_PAYMENTS)
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .stream()
        )
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            results.append(data)
        return results

    # ========== WHATSAPP STATS ==========
    def get_whatsapp_stats(self) -> List[Dict[str, Any]]:
        return self._collection_to_list(COLLECTION_WHATSAPP_STATS)

    # ========== SIDE HUSTLE ==========
    def get_side_hustle(self) -> List[Dict[str, Any]]:
        return self._collection_to_list(COLLECTION_SIDE_HUSTLE)

    # ========== HONEYPOT LOGS ==========
    def get_honeypot_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        docs = (
            self.db.collection(COLLECTION_HONEYPOT_LOGS)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            results.append(data)
        return results

    # ========== COMPETITION WINNERS ==========
    def get_competition_winners(self) -> List[Dict[str, Any]]:
        return self._collection_to_list(COLLECTION_COMPETITION_WINNERS)

    # ========== DAILY SNAPSHOTS ==========
    def get_daily_snapshots(self, limit: int = 30) -> List[Dict[str, Any]]:
        docs = (
            self.db.collection(COLLECTION_DAILY_SNAPSHOTS)
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            results.append(data)
        return results

    # ========== WRITE OPERATIONS ==========
    def save_snapshot(self, data: Dict[str, Any]) -> str:
        """Save a daily snapshot to Firestore. Returns document ID."""
        data["createdAt"] = firestore.SERVER_TIMESTAMP
        doc_ref = self.db.collection(COLLECTION_DAILY_SNAPSHOTS).document()
        doc_ref.set(data)
        return doc_ref.id

    def save_prediction(self, data: Dict[str, Any]) -> str:
        """Save a prediction result."""
        data["createdAt"] = firestore.SERVER_TIMESTAMP
        doc_ref = self.db.collection("predictions").document()
        doc_ref.set(data)
        return doc_ref.id

    def save_report(self, report_type: str, data: Dict[str, Any]) -> str:
        """Save a report (daily/weekly/monthly)."""
        data["type"] = report_type
        data["createdAt"] = firestore.SERVER_TIMESTAMP
        doc_ref = self.db.collection("reports").document()
        doc_ref.set(data)
        return doc_ref.id


# ========== MODULE-LEVEL SINGLETON ==========
_firebase_client: Optional[FirebaseClient] = None


def get_firebase_client() -> FirebaseClient:
    """Get or create the Firebase client singleton."""
    global _firebase_client
    if _firebase_client is None:
        _firebase_client = FirebaseClient()
    return _firebase_client