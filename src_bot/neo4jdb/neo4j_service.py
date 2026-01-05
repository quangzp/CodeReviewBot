from src_bot.neo4jdb.neo4j_db import Neo4jDB
from src_bot.neo4jdb.neo4j_dto import Neo4jNodeDto,Neo4jTraversalResultDto,Neo4jPathDto,Neo4jRelationshipDto
from typing import List,Optional

def _node_to_dto(node) -> Neo4jNodeDto:
    if not node:
        return None

    node_dict = dict(node)
    return Neo4jNodeDto(
        id=node.id,
        labels=list(node.labels),
        properties=node_dict,
        **node_dict
    )
def _path_to_dto(path) -> Optional[Neo4jPathDto]:
    if not path:
        return None

    nodes = [_node_to_dto(node) for node in path.nodes]
    relationships = []
    path_summary = []

    for i, rel in enumerate(path.relationships):
        start_node, end_node = _get_relationship_nodes(nodes, i)

        rel_data = _create_relationship_data(rel, start_node, end_node)
        relationships.append(rel_data)

        summary_item = _create_summary_item(i, rel, start_node, end_node)
        path_summary.append(summary_item)
    return Neo4jPathDto(
        start_node=nodes[0] if nodes else None,
        end_node=nodes[-1] if nodes else None,
        total_length=len(relationships),
        nodes=nodes,
        relationships=relationships,
        path_summary=path_summary
    )

def _get_relationship_nodes(nodes, index):
    start_node = nodes[index] if index < len(nodes) else None
    end_node = nodes[index + 1] if index + 1 < len(nodes) else None
    return start_node, end_node


def _create_relationship_data(rel, start_node, end_node):
    return {
        "type": rel.type,
        "start_node": start_node,
        "end_node": end_node,
        "properties": dict(rel)
    }


def _create_summary_item(step_index, rel, start_node, end_node):
    return {
        "step": step_index + 1,
        "from": _create_node_summary(start_node) if start_node else None,
        "relationship": rel.type,
        "to": _create_node_summary(end_node) if end_node else None
    }


def _create_node_summary(node):
    return {
        "class_name": node.class_name if node else None,
        "method_name": node.method_name if node else None,
        "node_type": node.labels[0] if node and node.labels else None
    }

class Neo4jService:
    def __init__(self, db: Neo4jDB | None = None):
        self.db = db or Neo4jDB()
    def get_node_by_ast_hash(self, ast_hash: str) -> Optional[Neo4jNodeDto]:
        query = """
        MATCH (n) WHERE n.ast_hash = $ast_hash RETURN n
        """
        with self.db.driver.session() as session:
            result = session.run(query, {"ast_hash": ast_hash}).single()
            return _node_to_dto(result["n"]) if result else None
        
    def get_related_nodes(
            self,
            target_nodes: List[Neo4jNodeDto],
            max_level: int = 20,
            min_level: int = 1,
            relationship_filter: str = "CALL>|<IMPLEMENT|<EXTEND|USE>|<BRANCH"
    ) -> List[Neo4jTraversalResultDto]:
        query = """
        WITH $targets AS targets
        MATCH (endpoint)
        WHERE endpoint.project_id = $targets[0].project_id
        AND any(t IN targets WHERE
          t.class_name = endpoint.class_name AND
          t.branch = endpoint.branch AND
          (
            (t.method_name IS NULL AND endpoint.method_name IS NULL)
            OR (t.method_name = endpoint.method_name)
          )
        )
        CALL apoc.path.expandConfig(endpoint, {
          relationshipFilter: "CALL>|<IMPLEMENT|<EXTEND|USE>|<BRANCH",
          minLevel: $min_level,
          maxLevel: $max_level,
          bfs: true,
          uniqueness: "NODE_GLOBAL",
          filterStartNode: false
        }) YIELD path
        WITH endpoint, path,
             nodes(path) AS node_list,
             relationships(path) AS rel_list
        WITH endpoint, path, node_list, rel_list, 
            [i IN range(0, size(rel_list) - 1) |
            CASE 
                WHEN type(rel_list[i]) = 'BRANCH'
                    AND node_list[i + 1].branch = 'develop'
                    AND node_list[i].branch = 'main'
                THEN node_list[i+1]
                ELSE null
            END
            ] AS exclude_nodes
        WITH endpoint, path, node_list, rel_list, exclude_nodes,
             [i IN range(0, size(rel_list)-1) |
                CASE
                  WHEN type(rel_list[i]) = 'CALL'
                       AND node_list[i+1].method_name IS NOT NULL
                  THEN node_list[i+1]

                  WHEN type(rel_list[i]) IN ['IMPLEMENT', 'EXTEND', 'BRANCH']
                  THEN node_list[i+1]

                  WHEN type(rel_list[i]) = 'USE'
                       AND node_list[i+1].method_name IS NULL
                  THEN node_list[i+1]
                  ELSE null
                END
             ] AS filtered_nodes
        RETURN endpoint, path,
               [node IN filtered_nodes WHERE node IS NOT NULL AND NOT node IN exclude_nodes] AS visited_nodes
        ORDER BY path
        """

        params = {
            'targets': [node.model_dump() for node in target_nodes],
            'relationship_filter': relationship_filter,
            'min_level': min_level,
            'max_level': max_level
        }

        with self.db.driver.session() as session:
            result = session.run(query, params)
            return [
                Neo4jTraversalResultDto(
                    endpoint=_node_to_dto(record['endpoint']),
                    paths=_path_to_dto(record['path']),
                    visited_nodes=[_node_to_dto(node) for node in record['visited_nodes']]
                )
                for record in result
            ]
    
    def extract_relationships(
        traversal_results: List[Neo4jTraversalResultDto]
    ):
        results = []
        seen_relationships = set()

        def node_key(node: Neo4jNodeDto):
            return (
                node.id
                or node.ast_hash
                or (node.class_name, node.method_name, node.file_path)
            )

        def rel_key(rel: Neo4jRelationshipDto):
            return (
                rel.type,
                node_key(rel.start_node),
                node_key(rel.end_node),
            )

        for traversal in traversal_results:
            path = traversal.paths
        if path and hasattr(path, "relationships"):
            for rel in path.relationships:
                rk = rel_key(rel)
                if rk in seen_relationships:
                    continue

                seen_relationships.add(rk)
                results.append({
                    "type": "relationship",
                    "relationship_type": rel.type,
                    "from_labels": rel.start_node.labels,
                    "to_labels": rel.end_node.labels,
                    "from_content": rel.start_node.content,
                    "to_content": rel.end_node.content,
                })

        return results