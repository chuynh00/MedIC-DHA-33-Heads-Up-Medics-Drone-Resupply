from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from resupply_engine.config import Settings
from resupply_engine.ingest import DispatchIngestCoordinator, HttpJsonIngestAdapter
from resupply_engine.models import PlanDecisionRequest, PlanningRequest, RecalculatePlanRequest
from resupply_engine.packing import PackingError
from resupply_engine.service import DispatchPlanningService, build_service


def create_app(settings: Settings | None = None, service: DispatchPlanningService | None = None) -> FastAPI:
    planner_service = service
    ingest_coordinator: DispatchIngestCoordinator | None = None
    http_adapter = HttpJsonIngestAdapter()
    resolved_settings = settings or Settings()
    app = FastAPI(title="Medic Drone Resupply Engine", version="0.1.0")

    def get_service() -> DispatchPlanningService:
        nonlocal planner_service
        if planner_service is None:
            planner_service = build_service(resolved_settings)
        return planner_service

    def get_ingest_coordinator() -> DispatchIngestCoordinator:
        nonlocal ingest_coordinator
        if ingest_coordinator is None:
            ingest_coordinator = DispatchIngestCoordinator(get_service())
        return ingest_coordinator

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/plans")
    def create_plan(planning_request: PlanningRequest, request: Request):
        try:
            ingested_request = http_adapter.ingest(
                planning_request,
                route_path=str(request.url.path),
                client_host=request.client.host if request.client else None,
                content_type=request.headers.get("content-type"),
            )
            return get_ingest_coordinator().submit(ingested_request)
        except (ValueError, KeyError, PackingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/plans/{plan_id}")
    def get_plan(plan_id: str):
        plan = get_service().get_plan(plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="Plan not found")
        return plan

    @app.post("/v1/plans/{plan_id}/recalculate")
    def recalculate_plan(plan_id: str, request: RecalculatePlanRequest):
        try:
            return get_service().recalculate_plan(plan_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, PackingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/plans/{plan_id}/decision")
    def record_decision(plan_id: str, request: PlanDecisionRequest):
        try:
            return get_service().record_decision(plan_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
