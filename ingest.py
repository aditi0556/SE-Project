import asyncio
from pathlib import Path

from pypdf import PdfReader
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


QDRANT_URL = "http://localhost:6333"


FILES = {
    "healthcare_knowledge": Path("data/healthcare_db.pdf"),
    "sdlc_knowledge": Path("data/sdlc_knowledge.txt"),
}


def extract_text(file_path: Path) -> str:

    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")

    elif file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n\n"

        return text

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def chunk_text(text: str, chunk_size: int = 1000):

    paragraphs = text.split("\n\n")

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current) + len(paragraph) <= chunk_size:
            current += paragraph + "\n\n"

        else:
            if current:
                chunks.append(current.strip())

            current = paragraph + "\n\n"

    if current:
        chunks.append(current.strip())

    return chunks


async def ingest_file(collection_name: str, file_path: Path):

    print(f"\n{'=' * 60}")
    print(f"Collection : {collection_name}")
    print(f"File       : {file_path}")
    print(f"{'=' * 60}")

    # 1. Extract text
    text = extract_text(file_path)

    print(f"Extracted {len(text)} characters")

    # 2. Create chunks
    chunks = chunk_text(text)

    print(f"Created {len(chunks)} chunks")

    # 3. Start Qdrant MCP server for this collection
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-qdrant"],
        env={
            "QDRANT_URL": QDRANT_URL,
            "COLLECTION_NAME": collection_name,
            "EMBEDDING_PROVIDER": "fastembed",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # 4. Check available tools
            tools = await session.list_tools()

            print("\nAvailable tools:")

            for tool in tools.tools:
                print("-", tool.name)

            # 5. Store chunks
            for i, chunk in enumerate(chunks):

                print(f"Storing chunk {i + 1}/{len(chunks)}")

                result = await session.call_tool(
                    "qdrant-store",
                    {
                        "information": chunk
                    }
                )

                print(result)

    print(f"\nFinished ingesting {file_path}")


async def main():

    for collection_name, file_path in FILES.items():

        await ingest_file(
            collection_name,
            file_path
        )

    print("\n" + "=" * 60)
    print("ALL INGESTION COMPLETE")

    print("\nQdrant collections:")
    print("- healthcare_knowledge")
    print("- sdlc_knowledge")


if __name__ == "__main__":
    asyncio.run(main())