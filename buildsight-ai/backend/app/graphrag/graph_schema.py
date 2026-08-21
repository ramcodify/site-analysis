"""BuildSight AI — Knowledge Graph Schema Definitions

Defines all research-grade Entity Node Types and Relationship Types for the Construction Intelligence Knowledge Graph.
"""

from enum import Enum


class NodeType(str, Enum):
    PROJECT             = "Project"
    WORKER              = "Worker"
    WORKER_SESSION      = "WorkerSession"
    CAMERA              = "Camera"
    ZONE                = "Zone"
    CONSTRUCTION_STAGE  = "ConstructionStage"
    ACTIVITY            = "Activity"
    PPE_ITEM            = "PPEItem"
    VIOLATION           = "Violation"
    RISK_EVENT          = "RiskEvent"
    PROGRESS_RECORD     = "ProgressRecord"
    DELAY_PREDICTION    = "DelayPrediction"
    SCHEDULE_MILESTONE  = "ScheduleMilestone"
    SAFETY_RULE         = "SafetyRule"
    HAZARD              = "Hazard"
    REPORT              = "Report"
    KNOWLEDGE_DOCUMENT  = "KnowledgeDocument"
    KNOWLEDGE_CHUNK     = "KnowledgeChunk"


class RelationType(str, Enum):
    WORKED_IN           = "WORKED_IN"
    DETECTED_BY         = "DETECTED_BY"
    PERFORMED           = "PERFORMED"
    WEARING             = "WEARING"
    MISSING             = "MISSING"
    VIOLATED            = "VIOLATED"
    OCCURRED_IN         = "OCCURRED_IN"
    OCCURRED_AT         = "OCCURRED_AT"
    ASSOCIATED_WITH     = "ASSOCIATED_WITH"
    HAS_RISK            = "HAS_RISK"
    HAS_VIOLATION       = "HAS_VIOLATION"
    DURING_STAGE        = "DURING_STAGE"
    NEXT_STAGE          = "NEXT_STAGE"
    CAUSED_BY           = "CAUSED_BY"
    RELATED_TO          = "RELATED_TO"
    RECORDED_BY         = "RECORDED_BY"
    GENERATED_FROM      = "GENERATED_FROM"
    APPLIES_TO          = "APPLIES_TO"
    REQUIRES            = "REQUIRES"
    DESCRIBED_BY        = "DESCRIBED_BY"
    CONTAINS            = "CONTAINS"
    BEFORE              = "BEFORE"
    AFTER               = "AFTER"
    REPEATED_IN         = "REPEATED_IN"
    AFFECTS_PROGRESS    = "AFFECTS_PROGRESS"
    CONTRIBUTES_TO_DELAY= "CONTRIBUTES_TO_DELAY"
    PREDICTED_FOR       = "PREDICTED_FOR"
    BEHIND_SCHEDULE     = "BEHIND_SCHEDULE"
