import json


SYSTEM_PROMPT = """[Task Definition]
Execute a structured, four-stage visual reasoning analysis for the provided image. Proceed through each stage sequentially and encapsulate your outputs using the specified tags.

[Chain-of-Thought]
Stage 1: Object Category Detection
Task: Identify all unique object categories in the image.
Requirements: Only output categories that are clearly visible and identifiable in the image.Maintain no duplicates. Output Format: Output as a JSON array within <CATEGORY> and
</CATEGORY> tags.

Stage 2: Object Instance Grounding
Task: Detect and localize every individual instance of the categories identified in Stage 1 sequentially.
Requirements: Process categories in the order identified in Stage 1. The object names must strictly correspond to the categories identified
in Stage 1. Assign sequential instance numbers to objects within each category (e.g., man.1, man.2, car.1). Provide precise bounding
boxes in [x1, y1, x2, y2] format (integer coordinates). All categories listed in Stage 1 must have instances detected in Stage 2. Output
Format: Output as a JSON array within <OBJECT> and </OBJECT> tags.

Stage 3:Relation Clue Reasoning
Task: For each object pair that may have a relationship, analyze the spatial and visual cues, then infer the most likely predicate
type and predicate relation. The three relation categories are:Spatial Relations: analyze spatial and topological relationships between all object pairs.
Possessive Relations: analyze ownership, composition, and part-whole relationships between all object pairs.Interactive Relations: analyze action-oriented and functional relationships between all object pairs.
Requirements: Treat each object in Stage 2 as the subject and compare it against every other object in the scene to identify
meaningful object pairs. For each pair, analyze the spatial and visual cues first, then infer the final relation type and
predicate. All relations must be between objects localized in Stage 2. If an object pair has multiple valid relation types or
predicates, separate them with /. Output Format: Output as a JSON array within <CLUE> and</CLUE> tags.

Stage 4: Relation Extraction
Task: Integrate the relation-clue reasoning from Stage 3 and organize it into three relation categories, then provide the complete
relation triplets in JSON format. 
Requirement: Summarize the reasoning in Stage 3 into relation triplets in JSON format. Output Format: Output as a JSON array within <RELATION> and</RELATION> tags.

[In-Context Example]
Complete Output Example:
<CATEGORY>{"categories": [{"id": "light"}, {"id": "tire"}, {"id": "window"}, {"id": "truck"}, {"id": "man"}, {"id": "car"}]}</CATEGORY>
<OBJECT>{"objects": [{"id": "light.1", "bbox": [104, 119, 221, 160]}, {"id": "tire.1", "bbox": [474, 403, 820, 782]}, {"id": "window.1", "bbox": [143, 181, 188, 245]}, {"id": "truck.1", "bbox": [36, 94, 852, 768]}, {"id": "truck.2", "bbox": [29, 49, 964, 988]}, {"id": "man.1", "bbox": [211, 96, 482, 475]}, {"id": "car.1", "bbox": [82, 154, 996, 785]}]}</OBJECT>
<CLUE>
(light.1, truck.1): Spatial: The light is positioned on the upper-left front area of the truck and lies within the truck's overall bounds. Visual: A red emergency light is mounted on the top/front of the fire truck cab. Type: spatial/interaction/interaction. Final predicate: on/mounted on/attached to
(tire.1, truck.2): Spatial: The tire is located in the lower-right portion of the truck and is contained within the truck's area. Visual: A large wheel is clearly a component of the fire truck. Type: spatial. Final predicate: on
(man.1, truck.2): Spatial: The man is inside the left-central cab area of the truck, overlapping its doorway region. Visual: A firefighter is sitting in the cab/door opening of the fire truck. Type: spatial/interaction. Final predicate: on/standing on
(window.1, truck.2): Spatial: The window is located on the upper-left cab section and falls fully within the truck's outline. Visual: A cab window is visible as a built-in feature of the truck. Type: spatial. Final predicate: on
(truck.1, truck.2): Spatial: The smaller truck box lies almost entirely within the larger truck box, with strong overlap indicating the same vehicle. Visual: Both boxes describe the red fire truck, likely one tighter crop and one broader view of that vehicle. Type: possession. Final predicate: part of
(light.1, truck.2): Spatial: The light appears at the top-front edge of the truck, sitting directly on its roofline. Visual: The emergency beacon is fixed to the truck's cab roof. Type: possession. Final predicate: part of
(window.1, truck.1): Spatial: The window is embedded in the upper-left side of the truck cab. Visual: The visible side window belongs to the truck cab structure. Type: possession. Final predicate: part of
(tire.1, truck.1): Spatial: The tire sits along the truck's lower side, enclosed by and aligned with the vehicle's bottom body region. Visual: The visible wheel is a mounted truck wheel, clearly a component attached beneath the fire truck. Type: possession. Final predicate: part of
(man.1, truck.1): Spatial: The man is seated within the cab opening, above the step and inside the truck's body area. Visual: A firefighter is riding in the fire truck. Type: spatial. Final predicate: in
(car.1, truck.2): Spatial: The car occupies the lower-left foreground and closely overlaps the truck's vicinity, indicating immediate proximity between vehicles. Visual: A dark car mirror and door edge appear beside the large red fire truck, showing the car nearby. Type: spatial. Final predicate: near
</CLUE>
<RELATION>{"relations": {"spatial_relations": [{"subject": "light.1", "predicate": "on", "object": "truck.1"}, {"subject": "tire.1", "predicate": "on", "object": "truck.2"}, {"subject": "man.1", "predicate": "on", "object": "truck.2"}, {"subject": "window.1", "predicate": "on", "object": "truck.2"}, {"subject": "man.1", "predicate": "in", "object": "truck.1"}, {"subject": "car.1", "predicate": "near", "object": "truck.2"}], "possession_relations": [{"subject": "truck.1", "predicate": "part of", "object": "truck.2"}, {"subject": "light.1", "predicate": "part of", "object": "truck.2"}, {"subject": "window.1", "predicate": "part of", "object": "truck.1"}, {"subject": "tire.1", "predicate": "part of", "object": "truck.1"}], "interaction_relations": [{"subject": "light.1", "predicate": "mounted on", "object": "truck.1"}, {"subject": "light.1", "predicate": "attached to", "object": "truck.1"}, {"subject": "man.1", "predicate": "standing on", "object": "truck.2"}]}}</RELATION>
"""


USER_PROMPT = """Generate the four-stage analysis for the image using the specified <CATEGORY>, <OBJECT>, <CLUE>and <RELATION> tags."""
