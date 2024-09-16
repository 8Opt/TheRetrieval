from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

templates = Jinja2Templates(directory="templates")


from api.base import translate_to_en, talk_to_llm, generate_quote

base_route = APIRouter(
    prefix="/base"
)

@base_route.get("/", response_class=HTMLResponse)
async def get_landing_page(request: Request): 
    try: 
        quote = generate_quote()
    except: 
        quote = """"He who would learn to fly one day must first learn to stand and walk and run and climb and dance; one cannot fly into flying." — Nietzsche"""
    return templates.TemplateResponse(
        name="index.html", request=request, 
        context={
            "quote": quote
        } 
    )

@base_route.get("/translate/{input}")
async def translate(input:str): 
    try: 
        result = translate_to_en(input=input)
        print(result)
        return {"msg": result}
    except HTTPException as httpe: 
        raise HTTPException(status_code=404, 
                            detail="Get something wrong from the function") from httpe

@base_route.get('/llm/{prompt}')
async def llm_complete(prompt:str): 
    try: 
        result = talk_to_llm(prompt=prompt)
        return {"msg": result}
    except HTTPException as httpe: 
        raise HTTPException(status_code=404, 
                            detail="Get something wrong from the function") from httpe