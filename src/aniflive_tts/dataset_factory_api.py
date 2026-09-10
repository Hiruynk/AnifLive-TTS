from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NoReturn

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .dataset_factory import DatasetFactory, DatasetFactoryError, EnergyVadConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetIngestRequest(_StrictModel):
    sources: list[str] = Field(min_length=1, max_length=256)
    recursive: bool = True


class DatasetListImportRequest(_StrictModel):
    path: str = Field(min_length=1, max_length=4_096)


class DatasetAnnotationRequest(_StrictModel):
    transcript: str | None = Field(default=None, max_length=16_000)
    language: str | None = Field(default=None, max_length=80)
    speaker: str | None = Field(default=None, max_length=160)
    expression: str | None = Field(default=None, max_length=240)
    expression_intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)
    dominance: float | None = Field(default=None, ge=-1.0, le=1.0)
    style_description: str | None = Field(default=None, max_length=2_000)
    propagate: bool = True


class DatasetResampleRequest(_StrictModel):
    target_rate: int = Field(default=32_000, ge=8_000, le=192_000)
    mono: bool = True


class DatasetVadRequest(_StrictModel):
    frame_ms: int = Field(default=20, ge=5, le=100)
    hop_ms: int = Field(default=10, ge=1, le=100)
    threshold_dbfs: float = Field(default=-42.0, ge=-100.0, le=0.0)
    min_speech_ms: int = Field(default=120, ge=0, le=60_000)
    min_silence_ms: int = Field(default=180, ge=0, le=60_000)
    pad_ms: int = Field(default=40, ge=0, le=5_000)
    max_segment_ms: int = Field(default=15_000, ge=250, le=300_000)

    def config(self) -> EnergyVadConfig:
        return EnergyVadConfig(**self.model_dump()).validated()


class DatasetReviewRequest(_StrictModel):
    decision: str
    note: str = Field(default="", max_length=2_000)


class DatasetAnnotationVerificationRequest(_StrictModel):
    kind: str
    decision: str = "verified"
    note: str = Field(default="", max_length=2_000)


class DatasetSplitRequest(_StrictModel):
    train: float = Field(default=0.85, ge=0.0, le=1.0)
    validation: float = Field(default=0.1, ge=0.0, le=1.0)
    test: float = Field(default=0.05, ge=0.0, le=1.0)
    seed: str = Field(default="aniflive-tts-v1.4", min_length=1, max_length=200)


class DatasetStageRequest(_StrictModel):
    stage: str = Field(min_length=1, max_length=40)


class TargetSpeakerClipRecord(_StrictModel):
    path: str = Field(min_length=1, max_length=4_096)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    source_sha256: str = Field(min_length=64, max_length=64)
    source_path: str | None = Field(default=None, max_length=4_096)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    position: int = Field(ge=0)
    route: str = Field(default="review", min_length=1, max_length=40)
    speaker: dict[str, Any] = Field(default_factory=dict)
    acquisition: dict[str, Any] = Field(default_factory=dict)


class TargetSpeakerClipImportRequest(_StrictModel):
    records: list[TargetSpeakerClipRecord] = Field(min_length=1, max_length=100_000)
    parent_artifact_id: str | None = Field(default=None, max_length=100)


class DatasetFreezeRequest(_StrictModel):
    require_expressions: bool = True


def _raise_api_error(error: DatasetFactoryError) -> NoReturn:
    detail = str(error)
    status = 404 if "not found" in detail.lower() else 422
    raise HTTPException(status_code=status, detail=detail) from error


def _dataset_item(factory: DatasetFactory, dataset_id: str, item_id: str) -> dict[str, Any]:
    try:
        item = factory.get_item(item_id)
    except DatasetFactoryError as error:
        _raise_api_error(error)
    if item["dataset_id"] != dataset_id:
        raise HTTPException(status_code=404, detail="Dataset item was not found")
    return item


def create_dataset_factory_router(
    factory_provider: Callable[[], DatasetFactory],
    *,
    dataset_validator: Callable[[str], None] | None = None,
) -> APIRouter:
    """Create the Dataset Factory REST surface without owning application lifecycle."""

    router = APIRouter(prefix="/api/workstation/datasets", tags=["Dataset Factory"])

    def validate_dataset(dataset_id: str) -> None:
        if dataset_validator is None:
            return
        try:
            dataset_validator(dataset_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return factory_provider().capabilities()

    @router.get("/{dataset_id}/items")
    def list_items(
        dataset_id: str,
        kind: str | None = Query(default=None),
        review_status: str | None = Query(default=None),
        split_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            items = factory_provider().list_items(
                dataset_id,
                kind=kind,
                review_status=review_status,
                split_name=split_name,
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)
        return {"dataset_id": dataset_id, "items": items, "count": len(items)}

    @router.get("/{dataset_id}/state")
    def project_state(dataset_id: str) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().project_state(dataset_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.patch("/{dataset_id}/stage")
    def advance_stage(
        dataset_id: str, request: DatasetStageRequest
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().advance_project_stage(dataset_id, request.stage)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/import-target-speaker-clips")
    def import_target_speaker_clips(
        dataset_id: str, request: TargetSpeakerClipImportRequest
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().import_target_speaker_clips(
                dataset_id,
                [record.model_dump() for record in request.records],
                parent_artifact_id=request.parent_artifact_id,
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.get("/{dataset_id}/review-queue")
    def review_queue(dataset_id: str) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().review_queue(dataset_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.get("/{dataset_id}/expression-candidates")
    def expression_candidates(dataset_id: str) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().expression_reference_candidates(dataset_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/freeze")
    def freeze(dataset_id: str, request: DatasetFreezeRequest) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().freeze_dataset(
                dataset_id, require_expressions=request.require_expressions
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/ingest")
    def ingest(dataset_id: str, request: DatasetIngestRequest) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            items = factory_provider().ingest(
                dataset_id,
                tuple(Path(value) for value in request.sources),
                recursive=request.recursive,
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)
        return {"dataset_id": dataset_id, "items": items, "count": len(items)}

    @router.post("/{dataset_id}/import-list")
    def import_list(
        dataset_id: str,
        request: DatasetListImportRequest,
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().import_gpt_sovits_list(dataset_id, Path(request.path))
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.get("/{dataset_id}/items/{item_id}/audio")
    def audio(dataset_id: str, item_id: str) -> FileResponse:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            path = factory.verified_audio_path(item_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)
        return FileResponse(path, media_type="audio/wav")

    @router.get("/{dataset_id}/items/{item_id}/waveform")
    def waveform(
        dataset_id: str,
        item_id: str,
        bins: int = Query(default=640, ge=32, le=2_048),
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            return factory.waveform(item_id, bins=bins)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/items/{item_id}/quality")
    def quality(dataset_id: str, item_id: str) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            return factory.analyze_quality(item_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.patch("/{dataset_id}/items/{item_id}/annotations")
    def annotations(
        dataset_id: str,
        item_id: str,
        request: DatasetAnnotationRequest,
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        fields = request.model_fields_set - {"propagate"}
        if not fields:
            raise HTTPException(status_code=422, detail="At least one annotation field is required")
        values = {field: getattr(request, field) for field in fields}
        try:
            return factory.update_annotations(
                item_id,
                values,
                source="manual",
                propagate=request.propagate,
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/items/{item_id}/resample")
    def resample(
        dataset_id: str,
        item_id: str,
        request: DatasetResampleRequest,
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            return factory.resample(
                item_id,
                target_rate=request.target_rate,
                mono=request.mono,
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/items/{item_id}/vad")
    def analyze_vad(
        dataset_id: str,
        item_id: str,
        request: DatasetVadRequest,
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            return factory.analyze_vad(item_id, config=request.config())
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/items/{item_id}/segments")
    def segment(
        dataset_id: str,
        item_id: str,
        request: DatasetVadRequest,
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            items = factory.segment(item_id, config=request.config())
        except DatasetFactoryError as error:
            _raise_api_error(error)
        return {"dataset_id": dataset_id, "source_item_id": item_id, "items": items, "count": len(items)}

    @router.patch("/{dataset_id}/items/{item_id}/review")
    def review(
        dataset_id: str,
        item_id: str,
        request: DatasetReviewRequest,
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            return factory.review(item_id, decision=request.decision, note=request.note)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.get("/{dataset_id}/items/{item_id}/review-history")
    def review_history(dataset_id: str, item_id: str) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            history = factory.review_history(item_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)
        return {
            "dataset_id": dataset_id,
            "item_id": item_id,
            "count": len(history),
            "history": history,
        }

    @router.post("/{dataset_id}/items/{item_id}/annotation-verifications")
    def verify_annotation(
        dataset_id: str,
        item_id: str,
        request: DatasetAnnotationVerificationRequest,
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        _dataset_item(factory, dataset_id, item_id)
        try:
            return factory.verify_annotation(
                item_id,
                kind=request.kind,
                decision=request.decision,
                note=request.note,
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.get("/{dataset_id}/items/{item_id}/annotation-verifications")
    def annotation_verification_history(
        dataset_id: str,
        item_id: str,
        kind: str | None = Query(default=None),
    ) -> dict[str, Any]:
        validate_dataset(dataset_id)
        factory = factory_provider()
        item = _dataset_item(factory, dataset_id, item_id)
        try:
            history = factory.annotation_verification_history(item_id, kind=kind)
        except DatasetFactoryError as error:
            _raise_api_error(error)
        return {
            "dataset_id": dataset_id,
            "item_id": item_id,
            "count": len(history),
            "verification": item["verification"],
            "history": history,
        }

    @router.get("/{dataset_id}/review-summary")
    def review_summary(dataset_id: str) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().review_summary(dataset_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.post("/{dataset_id}/split")
    def assign_splits(dataset_id: str, request: DatasetSplitRequest) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().assign_splits(
                dataset_id,
                train=request.train,
                validation=request.validation,
                test=request.test,
                seed=request.seed,
            )
        except DatasetFactoryError as error:
            _raise_api_error(error)

    @router.get("/{dataset_id}/manifest")
    def manifest(dataset_id: str) -> dict[str, Any]:
        validate_dataset(dataset_id)
        try:
            return factory_provider().manifest(dataset_id)
        except DatasetFactoryError as error:
            _raise_api_error(error)

    return router
