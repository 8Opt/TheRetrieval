import os 
import json

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.api.searching import search_by_text

templates = Jinja2Templates(directory="templates")


search_route = APIRouter(
    prefix="/search"
)


@search_route.post("/")
async def search_text(request: Request): 
    try: 
        payload = await request.json()
        payload = json.loads(payload)
        query, top_k = payload['query'], payload['top_k']
        translator = request.app.state.translator
        vector_store = request.app.state.vector_store
        result = search_by_text(translator.run(query), top_k, vector_store)
        return Response(content=result, 
                        media_type="application/json", 
                        status_code=200)
    except HTTPException as httpe: 
        raise HTTPException(status_code=404, 
                            detail="Get something wrong from the function") from httpe

@search_route.get('/{image_idx}')
async def get_image(image_idx: str): 
    try: 
        image_type = image_idx.split('.')[-1]
        image_path = os.path.join('./db', image_idx)
        with open(image_path, "rb") as f:
            return Response(content=f.read(), 
                            media_type=f"image/{image_type}",
                            status_code=200)
    except HTTPException as httpe: 
        raise HTTPException(status_code=404, 
                            detail="Get something wrong from the function") from httpe