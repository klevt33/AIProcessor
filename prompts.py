"""
## Overview
The `Prompts` module provides a collection of prompt templates and utility methods for
interacting with fine-tuned language models (LLMs) and web search systems.
These prompts are designed to structure queries and responses for specific tasks such as
ranking, searching, and generating JSON outputs. Additionally, the module includes functionality
for preparing headers and payloads for API requests to LLMs.

"""

from utils import read_txt_file


class Prompts:
    """
    A class containing various prompt templates for interacting with fine-tuned LLMs and web search systems.
    These prompts are used to structure queries and responses for specific tasks such as
    ranking, searching, and generating JSON outputs.
    """

    @classmethod
    def get_fine_tuned_llm_prompt(cls):
        """
        Returns the prompt template for interacting with a fine-tuned LLM.

        The prompt instructs the LLM to format its response strictly as a JSON object with specific attributes.

        Returns:
            str: The fine-tuned LLM prompt template.
        """
        _FINE_TUNED_LLM_PROMPT = """
            {description}

            Format the response strictly as a JSON object. No additional text, explanations, or disclaimers - only return JSON in this structure:
            ```json
            {{
                "ManufacturerName": "string",
                "PartNumber": "string",
                "UNSPSC": "string",
            }}
            ```
            If any attribute is unavailable, return an empty string for that attribute."""
        return _FINE_TUNED_LLM_PROMPT

    @classmethod
    def get_fine_tuned_llm_prompt_for_unspsc(cls):
        """
        Returns the prompt template for interacting with a fine-tuned LLM only for UNSPSC code.

        The prompt instructs the LLM to format its response strictly as a JSON object with specific attribute.

        Returns:
            str: The fine-tuned LLM prompt template.
        """
        _FINE_TUNED_LLM_PROMPT_FOR_UNSPSC = """
            {description}

            Format the response strictly as a JSON object. No additional text, explanations, or disclaimers - only return JSON in this structure:
            ```json
            {{
                "UNSPSC": "string"
            }}
            ```
            If attribute is unavailable, return an empty string."""
        return _FINE_TUNED_LLM_PROMPT_FOR_UNSPSC

    @classmethod
    def get_web_results_ranking_system_prompt(cls):
        """
        Returns the system prompt for ranking web results.

        This prompt is used to instruct the system to compare and match electrical parts descriptions.

        Returns:
            str: The web results ranking system prompt.
        """
        return """
            You are a helpful assistant for comparing and matching electrical parts descriptions and generating structured JSON responses.
            """

    @classmethod
    def get_web_search_system_prompt(cls):
        """
        Returns the system prompt for web search.

        This prompt is used to instruct the system to find and structure information about electrical parts.

        Returns:
            str: The web search system prompt.
        """
        return """
            You are a helpful assistant to find and structure information about electrical parts.
            """

    @classmethod
    def get_web_results_ranking_prompt(cls, description, results_json):
        """
        Returns the prompt for ranking web results using an agent.

        Args:
            description (str): The input description to compare.
            results_json (str): The JSON array of potential matches.

        Returns:
            str: The web results ranking prompt.
        """
        _WEB_RESULTS_RANKING_PROMPT = f"""
            Input Description:
            {description}

            Web Grounding Info JSON:
            {results_json}
            Task:
            Compare the Input Description to each of the potential matches in the JSON array. Output the same JSON array with the following 5 attributes added to each of the array elements (keep existing attributes and their values): ManufacturerMatch, DetectedManufacturer, PartNumberMatch, DetectedPartNumber, ItemDescriptionMatch. Format the response strictly as a JSON with no additional text, explanations, or disclaimers.
            Assign the new property values based on the following principles.
            ManufacturerMatch:
            - Exact: The Input Description explicitly includes a manufacturer name, and it exactly matches the ManufacturerName JSON value, with only minor variations (e.g., "3M Electrical" vs. "3M" or “ABG BAG INC” vs “ABG BAG”). Use your general knowledge to identify equivalent manufacturer names.
            - Likely: ManufacturerName JSON value and Input Description manufacturer are a close but not definitive match due to abbreviation, partial listing, or minor inconsistencies (e.g., "Adrf" vs. " ADVANCED RF TECHNOLOGIES").
            - Possible: The Input Description manufacturer has some resemblance to the ManufacturerName JSON value but is uncertain (e.g., a similar-sounding name that could be a different company).
            - Mismatch: The manufacturer name in the Input Description is clearly different from the ManufacturerName JSON value.
            - Not Detected: The Input Description does not explicitly include a manufacturer name. Use this value if the Input Description lacks any mention of a manufacturer.
            DetectedManufacturer:
            - Empty string if ManufacturerMatch = “Not Detected”.
            - Otherwise, it should contain the Manufacturer Name from the Input Description. It should be extracted from the Input Description “as is” without any changes even if it's abbreviated or misspelled.
            PartNumberMatch:
            - No Match: The Input Description does not contain the PartNumber JSON value even considering variations in formatting.
            - Exact: The manufacturer part number in the Input Description exactly matches the PartNumber JSON value, without any differences.
            - Strong: The manufacturer part number in the Input Description matches the PartNumber JSON value except for minor formatting differences such as dashes, slashes, or spaces (e.g., "ABC12345" vs. "ABC-1234/5").
            - Partial: A significant portion of the part numbers match, but there may be a different prefix or suffix (e.g., "ABC123456" vs. "123456" or “XYZ789” vs “XYZ789-1”). This does not apply to cases where only a small fragment matches.
            - Possible: The part numbers share some similarities but lack enough certainty for a confident match.
            DetectedPartNumber:
            - Empty string if PartNumberMatch = “No Match”.
            - Otherwise, it should contain the Manufacturer Part Number from the Input Description that matches fully or partially the PartNumber JSON value. It should be extracted from the Input Description “as is” without any changes.
            ItemDescriptionMatch:
            - Exact: The ItemDescription JSON value is identical or nearly identical to the Input Description, with no substantive differences.
            - Very High: The ItemDescription JSON value closely matches the Input Description, with only minor variations in wording or format that do not change the meaning.
            - High: The ItemDescription JSON value matches key attributes of the Input Description but may have differences in phrasing, word order, or additional information.
            - Medium: The ItemDescription JSON value and the Input Description are somewhat related but have enough differences to introduce uncertainty. Some key details may be missing or inconsistent.
            - Low: The ItemDescription JSON value and the Input Description contain only partial similarities and lack clear alignment.
        """
        return _WEB_RESULTS_RANKING_PROMPT

    @classmethod
    def get_web_results_ranking_prompt_for_unspsc(cls, description, results_json):
        """
        Returns the prompt for ranking web results using an agent.

        Args:
            description (str): The input description to compare.
            results_json (str): The JSON array of potential matches.

        Returns:
            str: The web results ranking prompt.
        """
        _WEB_RESULTS_RANKING_PROMPT_FOR_UNSPSC = f"""
            Input Description:
            {description}

            Web Grounding Info JSON:
            {results_json}
            Task:
            Compare the Input Description to each of the potential matches in the JSON array. Output the same JSON array with the following attribute added to each of the array elements (keep existing attributes and their values): ItemDescriptionMatch. Format the response strictly as a JSON with no additional text, explanations, or disclaimers.
            Assign the new property values based on the following principles.
            ItemDescriptionMatch:
            - Exact: The ItemDescription JSON value is identical or nearly identical to the Input Description, with no substantive differences.
            - Very High: The ItemDescription JSON value closely matches the Input Description, with only minor variations in wording or format that do not change the meaning.
            - High: The ItemDescription JSON value matches key attributes of the Input Description but may have differences in phrasing, word order, or additional information.
            - Medium: The ItemDescription JSON value and the Input Description are somewhat related but have enough differences to introduce uncertainty. Some key details may be missing or inconsistent.
            - Low: The ItemDescription JSON value and the Input Description contain only partial similarities and lack clear alignment.
        """
        return _WEB_RESULTS_RANKING_PROMPT_FOR_UNSPSC

    @classmethod
    def get_web_search_prompt(cls, config, description):
        """
        Returns the prompt for performing a web search using an agent.

        Args:
            config (Config): Configuration object for the search.
            description (str): The input description to search for.

        Returns:
            str: The web search prompt.
        """
        _WEB_SEARCH_PROMPT = f"""
            Search the web to find electrical parts that match the following input description:
            {description}
            Perform exactly one Bing search request using the provided part description. Do not execute multiple search requests or variations.
            For each potential match, extract the following attributes from a single source (do not mix information from different websites): Manufacturer Name (also known as Brand, Maker, Producer), Manufacturer Part Number (also known as MPN, Part No.), which is the universal number from the original brand (e.g., Siemens, 3M), not a vendor-specific SKU or Item Number from a distributor (e.g., Grainger, Digi-Key), Item Description (also known as Product Name, Title), UPC Code (also known as Universal Product Code, U.P.C.), and UNSPSC Code (also known as Commodity Code, Classification Code).
            For each potential match, record the full Source URL exactly as shown in the browser's address bar, including protocol (https://), complete domain and subdomain, full path, and any query parameters or fragments. Do not truncate, shorten, or omit any portion of the URL.
            Validate each URL by ensuring it returns a successful HTTP status (e.g., 200 OK) and actually leads to the item's product page.
            Prioritize results from the following websites. If any matches are found on these sites, rank them higher in the response:
            {read_txt_file(config.predefined_sites_filepath)}
            Include the Priority Match attribute (Boolean: true if the source is from the priority site list, otherwise false).
            Include a numeric ID attribute showing the element index in the array starting from 1.
            Format the response strictly as a JSON array with 5 top matches. No additional text, explanations, or disclaimers - only return JSON in this structure:
            ```json
            [
                {{
                "ID": 1,
                "ManufacturerName": "string",
                "PartNumber": "string",
                "ItemDescription": "string",
                "UPC": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": true,
                }},
                {{
                "ID": 2,
                "ManufacturerName": "string",
                "PartNumber": "string",
                "ItemDescription": "string",
                "UPC": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }},
                {{
                "ID": 3,
                "ManufacturerName": "string",
                "PartNumber": "string",
                "ItemDescription": "string",
                "UPC": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }},
                {{
                "ID": 4,
                "ManufacturerName": "string",
                "PartNumber": "string",
                "ItemDescription": "string",
                "UPC": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }},
                {{
                "ID": 5,
                "ManufacturerName": "string",
                "PartNumber": "string",
                "ItemDescription": "string",
                "UPC": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }}
            ]```
            If any attribute is unavailable, return an empty string for that attribute.
            If multiple sources provide the same attribute values, include all of them as separate entries. Do not exclude lower confidence matches from Bing search results. Provide all 5 matches whenever possible.
            If you cannot find any relevant matches for the given description, you MUST return an empty JSON array: []. Do not return any other text or explanation.
        """
        return _WEB_SEARCH_PROMPT

    @classmethod
    def get_web_search_prompt_for_unspsc(cls, config, description):
        """
        Returns the prompt for performing a web search using an agent.

        Args:
            config (Config): Configuration object for the search.
            description (str): The input description to search for.

        Returns:
            str: The web search prompt.
        """
        _WEB_SEARCH_PROMPT_FOR_UNSPSC = f"""
            Search the web to find electrical parts that match the following input description:
            {description}
            Perform exactly one Bing search request using the provided part description. Do not execute multiple search requests or variations.
            For each potential match, extract the following attributes from a single source (do not mix information from different websites): Item Description (also known as Product Name, Title), and UNSPSC Code (also known as Commodity Code, Classification Code).
            For each potential match, record the full Source URL exactly as shown in the browser's address bar, including protocol (https://), complete domain and subdomain, full path, and any query parameters or fragments. Do not truncate, shorten, or omit any portion of the URL.
            Validate each URL by ensuring it returns a successful HTTP status (e.g., 200 OK) and actually leads to the item's product page.
            Prioritize results from the following websites. If any matches are found on these sites, rank them higher in the response:
            {read_txt_file(config.predefined_sites_filepath)}
            Include the Priority Match attribute (Boolean: true if the source is from the priority site list, otherwise false).
            Include a numeric ID attribute showing the element index in the array starting from 1.
            Format the response strictly as a JSON array with 5 top matches. No additional text, explanations, or disclaimers - only return JSON in this structure:
            ```json
            [
                {{
                "ID": 1,
                "ItemDescription": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": true,
                }},
                {{
                "ID": 2,
                "ItemDescription": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }},
                {{
                "ID": 3,
                "ItemDescription": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }},
                {{
                "ID": 4,
                "ItemDescription": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }},
                {{
                "ID": 5,
                "ItemDescription": "string",
                "UNSPSC": "string",
                "Source": "string",
                "PriorityMatch": false,
                }}
            ]```
            If any attribute is unavailable, return an empty string for that attribute.
            If multiple sources provide the same attribute values, include all of them as separate entries. Do not exclude lower confidence matches from Bing search results. Provide all 5 matches whenever possible.
            If you cannot find any relevant matches for the given description, you MUST return an empty JSON array: []. Do not return any other text or explanation.
        """
        return _WEB_SEARCH_PROMPT_FOR_UNSPSC

    @classmethod
    def get_context_validator_prompt(cls, invoice_text, mfr, pn, db_description):
        """
        Returns the prompt for the LLM Context Validator.

        Args:
            invoice_text (str): The original, unstructured invoice line item text.
            mfr (str): The manufacturer name from the matched database record.
            pn (str): The part number from the matched database record.
            db_description (str): The official product description from the matched database record.

        Returns:
            str: The formatted prompt to be sent to the LLM.
        """
        return f"""
        You are a procurement expert verifying if a database match is contextually correct.

        Input Invoice Text: "{invoice_text}"
        Matched Database Item: Manufacturer="{mfr}", PartNumber="{pn}", Description="{db_description}"

        Task: Identify the PRIMARY ITEM being sold in the invoice text and determine its relationship to the Matched Database Item.

        Select one of the following Context Types:
        1. DIRECT_MATCH: The invoice describes the Matched Item exactly.
           * SUBSTITUTION RULE: If the invoice mentions "Sub for", "Replaces", or "Alternative to", AND the Matched Item is the *new* item being supplied (not the reference item), this is a DIRECT_MATCH.
           * UOM RULE: Ignore differences in package quantity (e.g., "Box" vs "Each") if the core product identity is the same.
        2. REPLACEMENT_PART: The invoice item is a component, spare part, or repair item intended FOR the Matched Item.
           * GRAMMAR RULE: Look for phrases like "for", "fits", "used with", or "compatible with". If the Match is the item *after* these words, it is a REPLACEMENT_PART (e.g., "Blade FOR [Saw]").
        3. ACCESSORY_PART: The invoice item is an optional add-on for the Matched Item (e.g., "Case", "Charger", "Strap").
        4. LOT_OR_KIT: The invoice describes a bundle, kit, or assembly, but the Matched Item is only one component of that group.
        5. UNRELATED: The match is completely wrong or refers to a different product mentioned in the text (e.g., the item being replaced).
        """

    def get_header_and_payload_for_aoi(self, query):
        """
        Prepares the header and payload for fetching a response from the LLM.

        Args:
            query (str): The query to send to the LLM.

        Returns:
            tuple: A tuple containing the headers and payload for the request.
        """
        headers = {"Content-Type": "application/json", "api-key": self.CONFIG.AOAI_BASE_LLM_OPENAI_API_KEY}

        # Payload for the request
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an AI assistant that helps people find information."}],
                },
                {"role": "user", "content": [{"type": "text", "text": query}]},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800,
        }
        return headers, payload
