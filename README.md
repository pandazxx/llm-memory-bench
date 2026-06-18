# llm-memory-bench


# Purpose

A framework for LLM agent memory experiments. It's a framework that:
1. Common workflow for standard Memory construction, Memory retrieval steps
2. Plugin different models(systems)
3. Dataset management
4. Parameters management (Parameters that feed to each systems for tuning)
5. Standard visualizable output for each step for diagnosis
6. Experiment result comparison and visualization

## Data structure / concepts

**System**
- program of the model that to be plug into the framework
- Provide main function: Memory construction, Memory retrieval
- Accept set of parameter to fine-tune the model (__optional__)
- Manage the generated memory storage
- Render visualization of the memory constructed
- Render visualization of memory retrieval trail

**Experiment**
- System
- Set of parameters
- Dataset
- Test-set

**Dataset**
List of statements, which contains:
- Statement id: unique id, in form of S00001
- Statement: plain text
- Timestamp: when this statement happens
- Comment: purpose / remarks, only for human reference, ignored by system

**Test-set**
List of tests, which contains:
- Query Id: unique id, in form of Q00001
- Query: plain text
- Required retrieval: Statements should be retrieved for this query. Judgemental factor of the test result.
- Expected answer: Expected LLM answer
- Comment: purpose / remarks, only for human reference, ignored by system

**Query result**
Result of `Memory retrieval`, contains:
* Query id
* Retrieved statements: list of statement id retrieved for this query. To be compared with `Required retrieval` for result scoring
* Retrieval trail: model specific retrieval trail. Eg. Step 1. Top-k seeds, Step2, PPR matrix. Use this to render result visualization
* Answer: Plain text LLM answer

## Visualizations

**Memory Structure**

Model specific
- Table of the memory nodes, example structure:
	- Statement id
	- tags
	- context
- Diagram of the graph, example structure
	- Memory node: link to the entry of the memory node list
	- edges: with description and weight
	- Side display of detail on click

**Query result**
- Table of query result 
- Diagram of the trail of each query, example structure:
	- Seed nodes
	- Linkage retrieval
	- Side display of detail on click

## Test result storage structure
- `<dataset_name>_<testset_name>/`
	- `<system_name>_<parameterset_name>/`
		- `memory/`
			- `index.html`: visualization
			- `other assets files`
		- `result/`
			- `index.html`: visualization
			- `result.json`: result
	- `<index.html>`: comparison result of each system+parameterset

## Startup model implementations
- A-Mem
- A-Mem (late chunking) — A-Mem with corpus-level late-chunked embeddings for neighbour search
- HippoRAG
- HippoRAG2

reference project: https://github.com/pandazxx/a-mem-reproduction
Start with the demos under `experiments/systems` of the project


