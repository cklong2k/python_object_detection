from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

class QdrantCRUD:
    def __init__(self, host="localhost", port=6333, collection_name="my_collection"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, vector_size, distance="Cosine"):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={"size": vector_size, "distance": distance}
        )

    def insert_point(self, point_id, vector, payload=None):
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def get_point(self, point_id):
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id]
        )

    def update_point(self, point_id, vector=None, payload=None):
        if vector is not None:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)]
            )
        elif payload is not None:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[point_id]
            )

    def delete_point(self, point_id):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"points": [point_id]}
        )

    def search(self, vector, top=5, filter_payload=None):
        filter_obj = None
        if filter_payload:
            filter_obj = Filter(
                must=[
                    FieldCondition(
                        key=k,
                        match=MatchValue(value=v)
                    ) for k, v in filter_payload.items()
                ]
            )
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top,
            query_filter=filter_obj
        )