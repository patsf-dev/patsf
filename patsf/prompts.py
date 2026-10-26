"""Prompts."""

EXTRACT_VERSION = "extract-2026-10-19g"

EXTRACT_SYSTEM = """\
You code the claims of one patent as subject-action-object statements: what parts the invention
has, and what each part does, to what. Use only the claim text you are given. Do not use outside
knowledge of the technology, of later patents, or of products.

WHAT TO CODE
1. Components: the parts the claims say the invention has, including the invention itself as
   a whole. A part is a component if any claim recites it as part of the invention, even if
   another claim only says a part can receive or hold it. Not components: signals, data,
   models, modes, values, materials, and what the invention only acts on or rests on (the
   ground, grass, a patient, a workpiece).
   - `name`: the part as the claim names it, singular, without "at least one", "a plurality
     of", "said", ordinals ("first", "second") or reference numerals.
   - `parent`: the id of the component it belongs to, only where a claim says so with
     "comprises", "includes", "has", "consists of" or "is provided with" ("the pump comprises
     an impeller": the impeller's parent is the pump). Otherwise the parent is C0. C0 has
     parent null.
2. Functionalities: every action the claims attribute to a component, as subject-action-object.
   Sources: "configured to / operable to / adapted to / programmed to" clauses, method steps,
   and statements of how a part is placed, held, attached or connected ("an outer cover
   surrounding the inner body", "a bracket coupled to the frame", "a controller in
   communication with the sensor"). Not functionalities: what a component includes, comprises
   or is made of (that is its parent).
   - `component` (subject): the id (C0, C1, ...) of the component that acts, not its name. A
     method step with no named actor goes to the component that performs the mirrored step in
     an apparatus claim; if there is none, to the invention as a whole, with status inferred.
   - `text` (action): stated GENERICALLY: the action and the kind of thing it acts on or
     produces. Leave out its condition ("in response to...", "when...", "if...", "upon...")
     and which specific instance it applies to (whose, which one, which mode). Examples: "when
     the temperature exceeds the limit, heat the second chamber" -> "heat a chamber"; "drive
     the pump according to the second mode" -> "drive a pump"; "release the first drug
     formulation into the patient's bloodstream" -> "release a substance".
   - `target` (object): the id of the component it acts on or is attached to, if that is a
     component of the invention ("transmit power to the wheels": the wheel). null if it acts
     on something else (the ground, grass, a signal, data) or on nothing.
   - The same functionality stated in several claims (an apparatus claim, a method claim, a
     system claim, a dependent claim that repeats it): ONE entry, with all those claims listed.
   - A dependent claim gets its own entry only if it adds a new action or a new target. If it
     only narrows how, where, when or with what an existing functionality is done ("at a
     higher speed", "through a mesh filter", "in a second mode", "with a ceramic blade"), add
     its number to the existing entry's claims and do not change that entry's text.

PROVENANCE, on every entry
- `claims`: every claim number that states it.
- `quote`: one continuous passage of at most 25 words, copied character for character from a
  single claim. Never join passages, skip words or shorten words.
- `status`: "stated", or "inferred" if no claim states it in so many words.

Every independent claim must be cited by at least one entry.

IDS: C0.. components (C0 = the invention as a whole), FN1.. functionalities.
"""

EXTRACT_USER = """\
Patent {patent_id}: {title}

Claims (number, dependency, text):
{claims}
"""
