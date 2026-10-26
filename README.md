# Patent Structure & Functionality (`patsf`)

Generate an invention's components, structure, components' functionalities and their relations from the patent claims. 

| Step | Command | What it does |
|---|---|---|
| Fetch | `patsf fetch` | Fetch patent claims, dependencies, citations and dates from Google Patents, one patent at a time |
| Extract | `patsf extract` | Extracts the structure from each patent's claims: its components, how they are nested, each component's functionalities, and whether those functionalities links to other components in the invention. Only this step uses an LLM. |
| Architecture | `patsf architecture` | Based on extracted structure of an invention, measure of the invention's structure (hierarchy breadth / depth, link density, modularity, centralization, number of functionalities per component, etc.). |
| Compare | `patsf compare` | Based on comparing an invention's structure against its cited patents' structures, measure the extent to which components are new / re-combined, and the extent to which functionalities and links are new.|
| Agreement | `patsf agree` | Based on comparing two models' (or a model and the hand coding) extraction of structure, generate an agreement score based on F1 |

All steps read from / output into one SQLite database (`patsf.db`).

## Install

```bash
pip install -e .                    # includes the local-model client and the embedding library
pip install -e ".[anthropic]"       # only for using Claude's Sonnet 5.5 as a reference model
pip install -e ".[rules]"           # only for the rule-based baseline (spaCy)
```

## Models

Models are named profiles in `pyproject.toml`. `--use <name>` picks one (the `default` is used
otherwise). 
Options: 
- gpt-oss-20b (default), 
- qwen35-9b (alternative open weight model from a different family),
- claude (proprietary model. Sonnet 5.5), 
- rules (non-LLM baseline)

Run patsf from the folder that has `pyproject.toml`. 

### Running gpt-oss-20b with Ollama

gpt-oss-20b and qwen35-9b runs locally under [Ollama](https://ollama.com).

1. Install Ollama.
2. Download the models: `ollama pull gpt-oss:20b` and, for agreement, `ollama pull qwen3.5:9b`.
3. Start the server with a long enough context. Extraction prompts run to about 10–30k tokens.

Windows (PowerShell):

```powershell
Get-Process ollama* | Stop-Process          # stop the tray copy if it is running
$env:OLLAMA_CONTEXT_LENGTH = "32768"
ollama serve                                # leave this window open; then open another terminal for patsf
```

**Hardware.**  gpt-oss-20b takes ~5 min per patent for extraction on a 12 GB GPU with part of the model in system RAM. 
qwen35-9b runs fully in GPU and would take ~2-3 min per patent. claude and rules runs quickly (seconds per patent).

## Fetch patents and extract structure

```bash
patsf fetch US11059478                      # Fetch data for patent US11059478,
patsf --use qwen35-9b extract US11059478    # Extract the structure of US11059478, using Qwen3.5:9b. See Models for other options. 
```

`fetch` downloads `patents.google.com/patent/<id>/en`, caches the raw HTML (`html_cache/`) and parses it:
- claims: one line per claim;
- the claim tree comes from `claim-ref` links, or from the text ("according to claim 1");
- claim type (apparatus, method, medium, composition) comes from the root claim's preamble text;
- stores only the first independent claim and the claims depending on it are stored, to avoid repetition 
  of components across a product claim, a system claim and a method claim.

`extract` sends an invention's claims to the model with the coding rules in `patsf/prompts.py`. 
- This step outputs a structure containing a list of components (each with its parent), and a list of functionalities (subject-action-target). 
- The structure is validated against the schema in `patsf/schema.py` (`python -m patsf.schema` to print).
- This step has a repair round upon encountering errors (e.g., a functionality pointing a subject to itself). 


### Rule-based baseline
`patsf --use rules extract US11059478` codes the structure by subject-action-object (SAO) extraction (`patsf/rules.py`), without an LLM. It looks for
- components: the subject and the object (compound nouns + head noun);
- a functionality: "`verb` a `object`", performed by the subject.


## Compare structure against hand-coding

```bash
patsf handcode US11059478                                  # writes handcode/<patent>.xlsx, for human coding
patsf import handcode/US11059478.xlsx                      # stores human handcoded xlsx under the tag "human"
patsf agree US10799093 US11059478 --tags human gpt-oss-20b # computes agreement between model against human-code (or another model), takes one or multiple patents
```

`agree` reports, for every pair of profiles, an F1 score. 
- Elements of the same type (components, functionalities) are matched one-to-one when they cite overlapping claims 
  and use similar wording. Links match when both of their ends are matched. 
- The matching maximizes the total score. F1 = 2 × matches / (elements of A + elements of B), per element type. 


## Generate downstream measures
```bash
patsf architecture US11059478        # Measures based on the extracted invention's structure
patsf compare US11059478             # Measures based on a comparison of the extracted invention's structure against its prior art (cited inventions') structures
```

`architecture` reads a patent's own structure and outputs the following measures

| Measure | Definition |
|---|---|
| `components`, `functionalities`, `links` | counts | 
| `depth` | levels of the part-of tree below C0 | 
| `branching` | mean number of direct parts per component that has parts | 
| `link_density` | links / possible pairs of parts | 
| `unlinked_share` | share of parts in no link | 
| `disconnected_groups` | groups of linked parts with no link between them (connected components of the link network; unlinked parts are not counted). Separate groups can also reflect claims that recite parts without stating how they connect | 
| `modularity` | Newman modularity of the clusters found by greedy modularity maximization (networkx) on the linked parts | 
| `degree_centralization`, `hub` | Freeman degree centralization, and the part in the most links | 
| `functionalities_per_component` | functionalities performed by parts / parts | 

`compare` pools the cited patents. Then compares an extracted invention's elements (components, functionalities, and links) against the pool.


| Measure | Definition |
|---|---|
| `new_component_share` | share of new components in the focal invention |
| `new_functionality_share` | share of new functionalities in the focal invention |
| `link_new_component_share` | share of links with at least one new component in the focal invention |
| `link_recombination_share` | share of links whose two components both exist in prior art, but are never joined in one cited patent |
| `component_recombination_share` | share of component pairs that no single cited patent holds together |
| `cited_compared` | number of US cited patents compared |

Output files in `results/<profile>/`: `<patent>.compare.patent.csv`, `.components.csv`,
`.functionalities.csv`, `.links.csv` and `.claims.csv`. Every row names its closest prior-art
element and the match score, so each value can be checked.

### Existing versus new

Elements are compared in two steps within `compare`, using a standard retrieve-then-rerank design:

1. **Shortlist.** Sentence embeddings from an open model trained on patents (PatentSBERTa) 
   give each prior-art element a cosine similarity to a focal invention element; 
   then `shortlist` identifies the most similar elements with at least `shortlist_floor`.
2. **Decide.** Then a cross-encoder reads the two texts together, scores each
   shortlisted pair from 0 to 1, and identifies matched pairs based on a threshold.

Models and settings are described in [tool.patsf.compare] `pyproject.toml`. They should be
calibrated against hand-coded patents before use. 

