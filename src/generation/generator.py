
from __future__ import annotations

import json                                         
import re                                          
from dataclasses import dataclass, field         
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import openai                                       

from src.config.settings import get_config, get_secrets 
from src.retrieval.retriever import ContextItem     
from src.utils.logger import get_logger             

logger = get_logger(__name__)



@dataclass
class GenerationResult:
    """The output of the generation stage, including the answer and provenance."""
    answer: str                                    
    citations: List[Dict[str, Any]] = field(default_factory=list)  
    sources: List[str] = field(default_factory=list)  
    faithfulness_score: Optional[float] = None      
    has_conflict: bool = False                   
    conflict_resolution: Optional[str] = None 
    model_used: str = ""                        
    prompt_tokens: int = 0                     
    completion_tokens: int = 0                 



class Generator:
    """
    Takes a query and retrieved context items and generates a grounded, cited answer.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._gen_cfg = cfg.generation              # Generation settings from YAML
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)


    def generate(
        self,
        query: str,
        context_items: List[ContextItem],
    ) -> GenerationResult:
        """
        Generate a grounded answer from the query and retrieved context.
        """

        has_conflict, conflict_note = self._detect_conflicts(context_items)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context_items, conflict_note)

        model = self._gen_cfg["model"]
        response = self._openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._gen_cfg["temperature"],
            max_tokens=self._gen_cfg["max_tokens"],
        )
        answer_raw = response.choices[0].message.content.strip()
        usage = response.usage                     

        answer_clean, citations = self._extract_citations(answer_raw, context_items)

        faithfulness_score: Optional[float] = None
        if self._gen_cfg["faithfulness_check"]["enabled"]:
            faithfulness_score = self._check_faithfulness(
                query=query,
                answer=answer_clean,
                context_items=context_items,
            )

        result = GenerationResult(
            answer=answer_clean,
            citations=citations,
            sources=list({item.chunk.source_file for item in context_items if item.chunk.source_file}),
            faithfulness_score=faithfulness_score,
            has_conflict=has_conflict,
            conflict_resolution=conflict_note,
            model_used=model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        logger.info(
            "Generation complete",
            extra={
                "faithfulness": faithfulness_score,
                "citations": len(citations),
                "has_conflict": has_conflict,
                "prompt_tokens": result.prompt_tokens,
            },
        )
        return result



    def _build_system_prompt(self) -> str:
        """
        System prompt enforcing grounded, citation-based generation.
        """

        return (
            "You are a precise, grounded question-answering assistant.\n"
            "Rules you MUST follow:\n"
            "1. Answer ONLY using the provided context passages. Do NOT use prior knowledge.\n"
            "2. For every claim you make, include an inline citation in the format [CITE:chunk_id].\n"
            "3. If the context does not contain enough information to answer, say: "
            "'I cannot answer this from the available context.'\n"
            "4. Do not speculate, infer, or add information not present in the context.\n"
            "5. If context passages contradict each other, acknowledge the contradiction "
            "and present both perspectives with their source dates."
        )

    def _build_user_prompt(
        self,
        query: str,
        context_items: List[ContextItem],
        conflict_note: Optional[str],
    ) -> str:
        """
        Build the user prompt with numbered context passages and citation instructions.
        """
        
        context_lines: List[str] = []
        for item in context_items:
            chunk = item.chunk
            header = (
                f"[chunk_id: {chunk.chunk_id or 'unknown'}] "
                f"[source: {chunk.source_name or 'unknown'}] "
                f"[date: {chunk.ingestion_ts or 'unknown'}]"
            )
            context_lines.append(f"{header}\n{chunk.text}")

        context_str = "\n\n---\n\n".join(context_lines) 

        prompt = f"Context passages:\n\n{context_str}\n\n"

        if conflict_note:
            prompt += f"Note: {conflict_note}\n\n" 

        prompt += f"Question: {query}\n\nAnswer (cite every claim with [CITE:chunk_id]):"
        return prompt



    def _detect_conflicts(
        self, context_items: List[ContextItem]
    ) -> Tuple[bool, Optional[str]]:
        """
        Lightweight LLM-based conflict detection across retrieved chunks.
        """
        if not self._gen_cfg["conflict_handling"]["detect_conflicts"]:
            return False, None                    

        if len(context_items) < 2:
            return False, None                    


        excerpts = [
            f"[{item.chunk.source_name or 'src'} / {item.chunk.ingestion_ts or '?'}]: "
            f"{item.chunk.text[:300]}"
            for item in context_items          
        ]
        excerpts_str = "\n\n".join(excerpts)

        prompt = (
            "Do the following text passages contain contradictory information?\n\n"
            f"{excerpts_str}\n\n"
            "Respond in JSON: {\"conflict\": true/false, \"description\": \"...\"}. "
            "Return only valid JSON."
        )
        try:
            response = self._openai.chat.completions.create(
                model="gpt-3.5-turbo",            
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)          
            if parsed.get("conflict"):
                desc = parsed.get("description", "Contradictory information found in sources.")
                logger.warning(f"Conflict detected: {desc}")
                return True, desc
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug(f"Conflict detection failed: {exc}")

        return False, None                      


    def _extract_citations(
        self,
        answer: str,
        context_items: List[ContextItem],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse [CITE:chunk_id] markers from the answer and resolve them to source metadata.
        """
        
        chunk_map: Dict[str, ContextItem] = {
            item.chunk.chunk_id: item
            for item in context_items
            if item.chunk.chunk_id
        }

        citation_pattern = re.compile(r"\[CITE:([^\]]+)\]")  # Match [CITE:abc123] markers
        found_ids = citation_pattern.findall(answer)       
        cleaned_answer = citation_pattern.sub("", answer).strip()

        citations: List[Dict[str, Any]] = []
        for cid in set(found_ids):            
            item = chunk_map.get(cid)
            citations.append({
                "chunk_id": cid,
                "source_file": item.chunk.source_file if item else None,
                "source_name": item.chunk.source_name if item else None,
                "ingestion_ts": item.chunk.ingestion_ts if item else None,
                "excerpt": item.chunk.text[:200] if item else None,
            })

        return cleaned_answer, citations


    def _check_faithfulness(
        self,
        query: str,
        answer: str,
        context_items: List[ContextItem],
    ) -> float:
        """
        Route to the configured faithfulness method (ragas or selfcheckgpt).
        """

        method = self._gen_cfg["faithfulness_check"]["method"]

        if method == "selfcheckgpt":
            return self._selfcheck_gpt_faithfulness(query, answer, context_items)
        else:
            return self._ragas_faithfulness(answer, context_items)

    def _ragas_faithfulness(
        self, answer: str, context_items: List[ContextItem]
    ) -> float:
        """
        Estimate faithfulness by extracting claims and verifying each against the context.
        """

        context_str = "\n".join(item.chunk.text[:300] for item in context_items)


        claims_prompt = (
            "Extract all factual claims from the following answer as a JSON array of strings. "
            "Return ONLY valid JSON, no prose.\n\n"
            f"Answer: {answer}"
        )
        try:
            claims_resp = self._openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": claims_prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            claims: List[str] = json.loads(claims_resp.choices[0].message.content.strip())
        except Exception:
            return 0.5                           

        if not claims:
            return 1.0                         


        grounded = 0
        for claim in claims:
            verify_prompt = (
                f"Context:\n{context_str}\n\n"
                f"Claim: {claim}\n\n"
                "Is this claim fully supported by the context? Answer YES or NO only."
            )
            try:
                verify_resp = self._openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": verify_prompt}],
                    temperature=0.0,
                    max_tokens=5,
                )
                verdict = verify_resp.choices[0].message.content.strip().upper()
                if verdict.startswith("YES"):
                    grounded += 1                  
            except Exception:
                pass                                

        score = grounded / len(claims)            
        logger.info(f"RAGAS faithfulness score: {score:.2f} ({grounded}/{len(claims)} claims grounded)")
        return score

    def _selfcheck_gpt_faithfulness(
        self,
        query: str,
        answer: str,
        context_items: List[ContextItem],
    ) -> float:
        """
        SelfCheckGPT faithfulness via N stochastic samples. High variance → low faithfulness.
        """

        n_samples = self._gen_cfg["faithfulness_check"]["selfcheck_samples"]  
        context_str = "\n".join(item.chunk.text[:200] for item in context_items)

        samples: List[str] = []
        for _ in range(n_samples):
            resp = self._openai.chat.completions.create(
                model=self._gen_cfg["model"],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
                        ),
                    }
                ],
                temperature=1.0,                  
                max_tokens=300,
            )
            samples.append(resp.choices[0].message.content.strip())

        
        consistencies: List[float] = []
        for sample in samples:
            agree_prompt = (
                f"Original answer: {answer}\n\n"
                f"Alternative answer: {sample}\n\n"
                "On a scale of 0.0 to 1.0, how consistent are these two answers in their factual claims? "
                "Return ONLY a float number."
            )
            try:
                agree_resp = self._openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": agree_prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                score_str = agree_resp.choices[0].message.content.strip()
                consistencies.append(float(score_str))
            except ValueError:
                consistencies.append(0.5)          

        avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0.5
        logger.info(f"SelfCheckGPT faithfulness: {avg_consistency:.2f}")
        return avg_consistency
