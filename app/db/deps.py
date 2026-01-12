"""
Dependencies for database and search operations.
"""
from app.db.client import search_client_manager
from azure.search.documents import SearchClient


def get_search_client() -> SearchClient:
    """
    Dependency function to get Azure Cognitive Search client.
    Used in FastAPI route dependencies.
    """
    return search_client_manager.get_client()
