# reference: https://github.com/langchain-ai/langchain/blob/master/cookbook/wikibase_agent.ipynb
import httpx
import json
import re
from mcp.server.fastmcp import FastMCP
from typing import List, Dict
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.utilities.wikidata import WikidataAPIWrapper

server = FastMCP("Wikidata MCP Server")
wikidata_query_tool = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

WIKIDATA_URL = "https://www.wikidata.org/w/api.php"
HEADER = {"Accept": "application/json", "User-Agent": "foobar"}


def _sanitize_summary(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


async def search_wikidata(query: str, is_entity: bool = True) -> str:
    """Search for a Wikidata item or property ID by its query."""
    if is_entity:
        try:
            results = wikidata_query_tool.api_wrapper.wikidata_mw.search(
                query, results=1
            )
        except Exception:
            return "Error searching Wikidata. Please try again later."
        if not results:
            return "No results found. Consider changing the search term."
        return results[0]

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 120,
        "srlimit": 1,
        "srqiprofile": "classic",
        "srwhat": "text",
        "format": "json",
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(WIKIDATA_URL, headers=HEADER, params=params)
            response.raise_for_status()
            title = response.json()["query"]["search"][0]["title"]
            title = title.split(":")[-1]
            return title
        except (httpx.HTTPError, KeyError, IndexError):
            return "No results found. Consider changing the search term."


@server.tool()
async def search_entity(query: str) -> str:
    """
    Search for a Wikidata entity ID by its query.

    Args:
        query (str): The query to search for. The query should be unambiguous enough to uniquely identify the entity.

    Returns:
        str: The Wikidata entity ID corresponding to the given query."
    """
    return await search_wikidata(query, is_entity=True)


@server.tool()
async def search_property(query: str) -> str:
    """
    Search for a Wikidata property ID by its query.

    Args:
        query (str): The query to search for. The query should be unambiguous enough to uniquely identify the property.

    Returns:
        str: The Wikidata property ID corresponding to the given query."
    """
    return await search_wikidata(query, is_entity=False)


@server.tool()
async def get_properties(entity_id: str) -> List[str]:
    """
    Get the properties associated with a given Wikidata entity ID.

    Args:
        entity_id (str): The entity ID to retrieve properties for. This should be a valid Wikidata entity ID.

    Returns:
        list: A list of property IDs associated with the given entity ID. If no properties are found, an empty list is returned.
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "claims",
        "format": "json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(WIKIDATA_URL, headers=HEADER, params=params)
    response.raise_for_status()
    data = response.json()
    return list(data.get("entities", {}).get(entity_id, {}).get("claims", {}).keys())


@server.tool()
async def execute_sparql(sparql_query: str) -> str:
    """
    Execute a SPARQL query on Wikidata.

    You may assume the following prefixes:
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>

    Args:
        sparql_query (str): The SPARQL query to execute.

    Returns:
        str: The JSON-formatted result of the SPARQL query execution. If there are no results, an empty JSON object will be returned.
    """
    url = "https://query.wikidata.org/sparql"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, params={"query": sparql_query, "format": "json"}
        )
    response.raise_for_status()
    result = response.json()["results"]["bindings"]
    return json.dumps(result)


@server.tool()
def wikidata_query(query: str) -> str:
    """Search Wikidata and return a summary for the given entity or QID."""
    try:
        summary = wikidata_query_tool.run(query)
    except Exception:
        return "Error querying Wikidata. Please try again later."
    if not isinstance(summary, str) or not summary.strip():
        return "No summary available."
    return _sanitize_summary(summary)


@server.tool()
async def get_metadata(entity_id: str, language: str = "en") -> Dict[str, str]:
    """
    Retrieve the English label and description for a given Wikidata entity ID.

    Args:
        entity_id (str): The entity ID to retrieve metadata for.
        language (str): The language code for the label and description (default is "en"). Use ISO 639-1 codes.

    Returns:
        dict: A dictionary containing the label and description of the entity, if available.
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels|descriptions",
        "languages": language,  # specify the desired language
        "format": "json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(WIKIDATA_URL, params=params)
    response.raise_for_status()
    data = response.json()
    entity_data = data.get("entities", {}).get(entity_id, {})
    label = (
        entity_data.get("labels", {}).get(language, {}).get("value", "No label found")
    )
    descriptions = (
        entity_data.get("descriptions", {})
        .get(language, {})
        .get("value", "No label found")
    )
    return {"Label": label, "Descriptions": descriptions}


if __name__ == "__main__":
    server.run()
