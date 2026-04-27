
import hashlib
from typing import Any, Callable, Dict, List, Optional, TypeVar

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.config.settings import get_config, get_secrets
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

class PIIGuard:
    """
    Scans text for personally identifiable information (PII) and redacts it.
    Applied at both ingestion time and before returning generated answers to users.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._pii_cfg = cfg.operations["pii"]
        self._enabled_scan = self._pii_cfg["detect_at_ingestion"]
        self._output_scan = self._pii_cfg["output_scanning"]
        self._entities = self._pii_cfg["entities_to_redact"] 

        if self._enabled_scan or self._output_scan:
            self._analyzer = AnalyzerEngine()  
            self._anonymizer = AnonymizerEngine() 
            logger.info("Presidio PII guard initialised")

    def redact(self, text: str, context: str = "ingestion") -> str:
        """
        Detect and redact PII from a text string.

        Args:
            text:    Text to scan.
            context: "ingestion" or "output" — controls which config flag is checked.

        Returns:
            Redacted text with PII entities replaced by their type tag (e.g. <PERSON>).
        """
        if context == "ingestion" and not self._enabled_scan:
            return text                             
        if context == "output" and not self._output_scan:
            return text                             

        try:
            analyzer_results = self._analyzer.analyze(
                text=text,
                entities=self._entities,            
                language="en",                      # Primary language (multilingual: use "auto")
            )

            if not analyzer_results:
                return text                      

            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
            )
            if len(analyzer_results) > 0:
                logger.info(
                    f"PII redacted in {context}: {len(analyzer_results)} entities removed",
                    extra={"entity_types": [r.entity_type for r in analyzer_results]},
                )
            return anonymized.text              

        except Exception as exc:
            logger.error(f"PII redaction failed: {exc}")                       
