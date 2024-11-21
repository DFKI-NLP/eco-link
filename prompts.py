query = """
"{component_name}" is a component in a manufactured product.
The product description is {product_description}.
The description of the material used to make {component_name} is {material}.
The manufacturer of {component_name} is {producer}.

What is the activity name and activity description in English for component "{component_name}"?

"""

response_example = """
======================================================

An example of a correctly formatted response:

Industrial activity name: C3 hydrocarbon production, mixture, petroleum refinery operation
Activity information: 
Gaseous mixture of C3-hydrocarbons, yielded from petroleum refinery operation, assumed to consists of 68% propene (also known as propylene or methyl ethylene) and 32% propane
Transformation process of crude oil entering the petroleum refinery ending with refinery products leaving the petroleum refinery.

======================================================

"""

search_context = """
======================================================

Web search results for search query "{search_query}":

{search_results}

======================================================

"""

instructions = """
======================================================

Instructions: 

You must answer correctly.
Identify the material and production process used to manufacture the component.
Provide the name of the industrial production activity in English for this material.
The component name and material may be in a language other than English, so translate it to English to understand the component and material.
Examples of production activities are "aluminium production, primary, ingot" or "acetaldehyde production, ethylene oxidation".
Following this, provide a one-sentence technical description of the material.
If unsure, make your best guess.
Do not provide any additional response. Any additional response beyond these items will be considered incorrect.

======================================================
"""

web_query_prompt = \
"""Instructions:

Formulate a query that can be entered into a search engine to identify the generic name of the material used to make {component_name}.
If information is unavailable, it is denoted as "unspecified".
Reply with the query text only. 
Do not provide any additional response. Any additional response beyond these items will be considered incorrect.

Examples of correct web search query responses:

================================

Question:
Component "SICHERUNGSBLECH" is a component used to make a manufactured product.
SICHERUNGSBLECH is made of material "S235JR".
The manufacturer of SICHERUNGSBLECH is "unspecified".

Response: s235jr material

================================

Question:
Component "SV-VE 60-12.0 rosa Platten,  8 mm" is a component used to make a manufactured product.
SV-VE 60-12.0 rosa Platten,  8 mm is made of material "unspecified".
The manufacturer of SV-VE 60-12.0 rosa Platten,  8 mm is "SV-Schaumstoffe GmbH".

Response: SV-VE 60-12.0 rosa Platten,  8 mm	SV-Schaumstoffe GmbH

================================

Question:
Component "Topf Oberteil" is a component used to make a manufactured product.
Topf Oberteil is made of material HC300.
The manufacturer of Topf Oberteil is unknown.

Response: material HC300

================================

Component "{component_name}" is a component used to make a manufactured product.
The description of the material used to make {component_name} is {material}.
The manufacturer of {component_name} is {producer}.

Response:"""

datasheet_context = """
======================================================

Following is a technical datasheet for "{component_name}":

{datasheet_content}

======================================================
"""