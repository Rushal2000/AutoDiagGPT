from chain import ask_with_fallback

test_queries = [
    {"query": "Hydraulic pump not building pressure", "system": "hydraulics"},
    {"query": "Engine oil pressure monitor light on", "system": "general"},
    {"query": "Machine travel speed is slow", "system": "hydraulics"},
    {"query": "Battery discharged, engine won't start", "system": "general"},
    {"query": "Error code E02 on monitor display", "system": "general"},
]

print("=" * 60)
print("  AutoDiagGPT — Pipeline Test")
print("=" * 60)

for i, test in enumerate(test_queries, 1):
    print(f"\nTest {i}/{len(test_queries)}")
    print(f"Query: {test['query']}")
    print(f"System: {test['system']}")
    print("-" * 60)

    result = ask_with_fallback(test["query"], system_filter=test["system"])
    print(result["answer"])
    print(f"\nProvider: {result['provider']}")
    print(f"Chunks retrieved: {result['num_chunks_retrieved']}")
    if result["sources"]:
        print("Sources:")
        for src in result["sources"]:
            print(f"   - {src['source']} (Page {src['page']}, {src['content_type']})")

print(f"\n{'=' * 60}")
print("All tests complete!")