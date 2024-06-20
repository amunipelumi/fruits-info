from fastapi import FastAPI
from .routers.post import router



app = FastAPI(project_name='Fruits-Info')
app.include_router(router)

@app.get('/')
def home_page():
    return {
        'message': 'hello world!!!'
    }