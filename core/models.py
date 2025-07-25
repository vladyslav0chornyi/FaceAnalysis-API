from insightface.app import FaceAnalysis
from openvino.runtime import Core

def init_face_analysis(provider='CPUExecutionProvider', ctx_id=0):
    face_app = FaceAnalysis(name='buffalo_l', providers=[provider])
    face_app.prepare(ctx_id=ctx_id)
    return face_app

def init_openvino_models(device_name="CPU"):
    core = Core()
    age_gender_model = core.read_model(model="models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml")
    age_gender_compiled = core.compile_model(model=age_gender_model, device_name=device_name)
    attr_model = core.read_model(model="models/intel/person-attributes-recognition-crossroad-0230/FP32/person-attributes-recognition-crossroad-0230.xml")
    attr_compiled = core.compile_model(model=attr_model, device_name=device_name)
    emotion_model = core.read_model(model="models/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml")
    emotion_compiled = core.compile_model(model=emotion_model, device_name=device_name)
    return {
        "core": core,
        "age_gender_model": age_gender_model,
        "age_gender_compiled": age_gender_compiled,
        "attr_model": attr_model,
        "attr_compiled": attr_compiled,
        "emotion_model": emotion_model,
        "emotion_compiled": emotion_compiled
    }