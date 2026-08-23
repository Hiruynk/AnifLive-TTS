from __future__ import annotations


class AnifLiveTTSError(RuntimeError):
    code = "ANIFLIVE_TTS_ERROR"


class ModelInspectionError(AnifLiveTTSError):
    code = "MODEL_INSPECTION_FAILED"


class EngineRebuildRequired(AnifLiveTTSError):
    code = "ENGINE_REBUILD_REQUIRED"


class PackageValidationError(AnifLiveTTSError):
    code = "PACKAGE_VALIDATION_FAILED"


class ExpressionNotImplemented(AnifLiveTTSError):
    code = "expression_not_implemented"

