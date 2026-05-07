
from __future__ import annotations

import re                                           
from dataclasses import dataclass, field            
from datetime import datetime                    
from typing import Any, Dict, List, Optional, Tuple

import spacy                                    
import openai                                     
from langdetect import detect as _langdetect    

from src.config.settings import get_config, get_secrets 
from src.utils.logger import get_logger     

logger = get_logger(__name__)



@dataclass
class ProcessedQuery:
    """
    The output of the query understanding stage.
    Carries the rewritten query, sub-questions, and metadata filters.
    """
    original_query: str                             
    expanded_query: str = ""                       
    standalone_query: str = ""                      
    sub_questions: List[str] = field(default_factory=list)  
    metadata_filters: Dict[str, Any] = field(default_factory=dict)  
    hypothetical_doc: Optional[str] = None          
    intent: Optional[str] = None                    
    language: Optional[str] = None          

    def final_query(self) -> str:
        """Return the best query string to use for retrieval."""
        return self.expanded_query or self.standalone_query or self.original_query



class QueryUnderstanding:
    """
    Applies the full query understanding pipeline to a raw user query.
    Each step is enabled/disabled via config.yaml.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._q_cfg = cfg.query                     
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)


        self._nlp: Optional[spacy.Language] = None
        if self._q_cfg["entity_recognition"]["enabled"]:
            try:
                self._nlp = spacy.load(
                    self._q_cfg["entity_recognition"]["model"]
                )
            except OSError:
                logger.warning(
                    "spaCy model not found. Run: python -m spacy download en_core_web_trf"
                )


    def process(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ProcessedQuery:
        """
        Runs the full query understanding pipeline.

        Steps include:
        - optional conversation condensation
        - intent classification
        - query expansion (e.g., HyDE)
        - sub-question decomposition
        - entity and metadata extraction
        - language detection
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        pq = ProcessedQuery(original_query=query)


        if self._q_cfg["conversation"]["enabled"] and conversation_history:
            pq.standalone_query = self._condense_with_history(query, conversation_history)
        else:
            pq.standalone_query = query            

        working_query = pq.standalone_query        

        pq.intent = self._classify_intent(working_query)

        if self._q_cfg["expansion"]["enabled"]:
            pq.expanded_query, pq.hypothetical_doc = self._expand_query(working_query)
        else:
            pq.expanded_query = working_query

        if self._q_cfg["decomposition"]["enabled"]:
            pq.sub_questions = self._decompose_query(working_query)

        if self._q_cfg["entity_recognition"]["enabled"]:
            pq.metadata_filters = self._extract_filters(working_query)

        try:
            pq.language = _langdetect(query)
        except Exception:
            pq.language = "en"

        logger.info(
            "Query processed",
            extra={
                "original": query,
                "standalone": pq.standalone_query,
                "intent": pq.intent,
                "sub_questions": len(pq.sub_questions),
                "filters": pq.metadata_filters,
            },
        )
        return pq


    def _condense_with_history(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """
        Turns a follow-up question into a standalone query by using chat history.

        Example:
            History: "Tell me about GDPR"
            Follow-up: "What are the penalties?"
            Output: "What are the penalties under GDPR?"
        """

        window = self._q_cfg["conversation"]["history_window"]   
        model = self._q_cfg["conversation"]["condense_model"]    

        trimmed = history[-window:]             

        history_str = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in trimmed
        )
        MAX_HISTORY_CHARS = 3_000
        if len(history_str) > MAX_HISTORY_CHARS:
            history_str = "..." + history_str[-MAX_HISTORY_CHARS:]

        system_prompt = (
            "You are a query reformulation assistant. "
            "Given a conversation history and a follow-up question, "
            "rewrite the follow-up as a complete, standalone question that "
            "includes all necessary context. Return ONLY the rewritten question."
        )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Conversation history:\n{history_str}\n\n"
                            f"Follow-up question: {query}\n\n"
                            "Standalone question:"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip() or query
        except Exception as exc:
            logger.warning("Condense-with-history failed (%s); using raw query.", exc)
            return query


    def _expand_query(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Expands short queries using techniques like HyDE.

        The idea is to generate a short hypothetical document that could answer
        the query, then use that for better retrieval coverage.

        Returns:
            (expanded_query, hypothetical_document)
        """

        method = self._q_cfg["expansion"]["method"]   # "hyde" | "query2doc" | "prf"

        if method == "hyde":
            model = self._q_cfg["expansion"]["hyde_model"]
            max_tok = self._q_cfg["expansion"]["hyde_max_tokens"]
            try:
                response = self._openai.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a document generation assistant. "
                                "Write a short, factual passage (2-4 sentences) that would "
                                "directly answer the following question. "
                                "Return ONLY the passage."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.3,
                    max_tokens=max_tok,
                )
                hypothetical_doc = response.choices[0].message.content.strip()
        
                return f"{query} {hypothetical_doc}", hypothetical_doc
            except Exception as exc:
                logger.warning("HyDE expansion failed (%s); using raw query.", exc)
                return query, None

        elif method == "query2doc":
            try:
                response = self._openai.chat.completions.create(
                    model=self._q_cfg["expansion"]["hyde_model"],
                    messages=[
                        {
                            "role": "user",
                            "content": f"Expand this search query into a detailed question: {query}",
                        }
                    ],
                    temperature=0.0,
                    max_tokens=100,
                )
                expanded = response.choices[0].message.content.strip()
                return f"{query} {expanded}", None
            except Exception as exc:
                logger.warning("query2doc expansion failed (%s); using raw query.", exc)
                return query, None

        else:
            return query, None


    def _classify_intent(self, query: str) -> str:
        """
        Classify the high-level intent of a query to guide downstream routing.
        """
        q_lower = query.lower()
        if any(kw in q_lower for kw in ["compare", "difference", "vs", "versus", "contrast"]):
            return "comparative"
        elif any(kw in q_lower for kw in ["list", "enumerate", "what are all", "give me"]):
            return "enumerative"
        elif any(kw in q_lower for kw in ["when", "date", "year", "time"]):
            return "temporal"
        elif any(kw in q_lower for kw in ["who", "person", "author", "ceo", "founder"]):
            return "entity"
        elif any(kw in q_lower for kw in ["why", "explain", "how does", "reason"]):
            return "analytical"
        else:
            return "factoid"                 
        

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex multi-intent query into atomic sub-questions.
        """
        max_sub = self._q_cfg["decomposition"]["max_sub_questions"]
        model = self._q_cfg["decomposition"].get(
            "model", self._q_cfg["conversation"]["condense_model"]
        )

        system_prompt = (
            f"You are a question decomposition assistant. "
            f"Break the following complex question into at most {max_sub} simpler, "
            f"self-contained sub-questions. Return ONLY a numbered list, one per line."
        )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Query decomposition failed (%s); skipping.", exc)
            return []

        sub_questions = []
        for line in raw.split("\n"):
            cleaned = re.sub(r"^\d+[.)]\s*", "", line.strip())
            if cleaned:
                sub_questions.append(cleaned)

        return sub_questions[:max_sub]


    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Run NER on the query to extract named entities and temporal references.
        These become metadata filters applied during vector store search.

        Example:
          "What did Apple announce in Q3 2023?" →
          {"entity_ORG": "Apple", "date_range": {"gte": "2023-07-01", "lte": "2023-09-30"}}
        """
        filters: Dict[str, Any] = {}

        if self._nlp:
            doc = self._nlp(query)            
            for ent in doc.ents:
                filter_key = f"entity_{ent.label_}"
                filters.setdefault(filter_key, []).append(ent.text)
                logger.debug("NER entity found: %s = '%s'", ent.label_, ent.text)

        if self._q_cfg["entity_recognition"]["temporal_grounding"]:
            date_filter = self._extract_date_filter(query)
            if date_filter:
                filters["date_range"] = date_filter

        return filters

    @staticmethod
    def _extract_date_filter(query: str) -> Optional[Dict[str, str]]:
        """
        Extract a date range filter from common temporal expressions in the query.
        Returns a dict with "gte" and "lte" ISO-8601 date strings, or None.
        """

        quarter_match = re.search(r"Q([1-4])\s+(\d{4})", query, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            quarter_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
            quarter_ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            return {
                "gte": f"{year}-{quarter_starts[quarter]}",
                "lte": f"{year}-{quarter_ends[quarter]}",
            }

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            year = year_match.group(1)
            return {"gte": f"{year}-01-01", "lte": f"{year}-12-31"}

        return None                            
