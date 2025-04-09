 评测脚本：
 python VLABench/tutorials/evaluation

 修改策略接口：
 已经实现客户端的接受
/mnt/data/310_jiarui/VLABench/VLABench/evaluation/model/policy/client.py

请在lerobot端部署服务端

实例：
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class PredictionRequest(BaseModel):
    observation: Dict[str, Any]
    parameters: Dict[str, Any]

class PredictionResponse(BaseModel):
    position: list
    euler: list
    gripper_state: float
    view_index: int

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 在这里实现你的预测逻辑
        # ...
        
        return {
            "position": [0.0, 0.0, 0.0],
            "euler": [0.0, 0.0, 0.0],
            "gripper_state": 0.0,
            "view_index": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))