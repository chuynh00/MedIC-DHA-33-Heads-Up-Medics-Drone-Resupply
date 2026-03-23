from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from resupply_engine.models import PlanningRequest, utc_now


TransportKind = Literal["http_json", "file_drop_json", "radio_gateway_text", "operator_entry"]


class IngestContext(BaseModel):
    transport: TransportKind
    adapter_name: str
    received_at: str = Field(default_factory=lambda: utc_now().isoformat())
    source_locator: str | None = None
    remote_host: str | None = None
    content_type: str | None = None


class IngestedPlanningRequest(BaseModel):
    planning_request: PlanningRequest
    ingest_context: IngestContext


class IngestAdapter(Protocol):
    def ingest(self, planning_request: PlanningRequest, **kwargs) -> IngestedPlanningRequest:
        ...


class HttpJsonIngestAdapter:
    adapter_name = "http_json_ingest_adapter"

    def ingest(
        self,
        planning_request: PlanningRequest,
        *,
        route_path: str,
        client_host: str | None,
        content_type: str | None,
    ) -> IngestedPlanningRequest:
        return IngestedPlanningRequest(
            planning_request=planning_request,
            ingest_context=IngestContext(
                transport="http_json",
                adapter_name=self.adapter_name,
                source_locator=route_path,
                remote_host=client_host,
                content_type=content_type,
            ),
        )


class FileDropJsonIngestAdapter:
    adapter_name = "file_drop_json_ingest_adapter"

    def ingest(
        self,
        planning_request: PlanningRequest,
        *,
        source_path: str,
    ) -> IngestedPlanningRequest:
        return IngestedPlanningRequest(
            planning_request=planning_request,
            ingest_context=IngestContext(
                transport="file_drop_json",
                adapter_name=self.adapter_name,
                source_locator=source_path,
                content_type="application/json",
            ),
        )


class DispatchIngestCoordinator:
    def __init__(self, planner_service) -> None:
        self.planner_service = planner_service

    def submit(self, ingested_request: IngestedPlanningRequest):
        return self.planner_service.create_plan(
            ingested_request.planning_request,
            ingest_context=ingested_request.ingest_context,
        )
