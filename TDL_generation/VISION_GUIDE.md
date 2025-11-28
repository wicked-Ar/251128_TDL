# Vision-Enhanced NL2TDL Pipeline

## 개요

MuJoCo 시뮬레이션 화면을 **Gemini Vision API**로 인식하여 물체의 위치와 상태를 파악하고, 이 정보를 활용하여 더 정확한 TDL을 생성하는 통합 파이프라인입니다.

---

## 아키텍처

```
[MuJoCo Scene]
     ↓ (capture)
[Scene Image]
     ↓ (Gemini Vision)
[Vision Analysis]
  - Object: apple
  - Position: center of table
  - Confidence: high
     ↓ (enhance)
[User NL] + [Vision Context]
     ↓ (Gemini LLM)
[TDL Code]
```

---

## 주요 기능

### 1. **자동 물체 인식**
- MuJoCo 렌더링 이미지에서 물체 검출
- 물체 종류, 위치, 상태 파악
- 신뢰도(confidence) 측정

### 2. **비전 컨텍스트 통합**
- 인식된 물체 정보를 TDL 생성 프롬프트에 추가
- LLM이 실제 장면 상황을 고려하여 TDL 생성
- 존재하지 않는 물체에 대한 명령 방지

### 3. **대화형 파이프라인**
- 실시간 장면 분석
- 자연어 명령 입력
- 즉시 TDL 생성 및 검증

---

## 파일 구조

```
TDL_generation/
├── vision_scene_analyzer.py           # 비전 분석 모듈
├── interactive_pipeline_with_vision.py # 통합 파이프라인
└── VISION_GUIDE.md                     # 이 문서
```

---

## 🚀 사용 방법

### 방법 1: MuJoCo 환경과 통합 (추천)

MuJoCo 시뮬레이션을 실행하면서 실시간으로 장면을 분석합니다.

```bash
cd C:/Users/Smart\ CPS/Desktop/xz/NL2TDL/TDL_generation

# MuJoCo 환경과 함께 실행
python interactive_pipeline_with_vision.py --mode mujoco
```

**대화형 명령어**:
```
> analyze                          # 현재 장면 분석
> generate pick the apple          # TDL 생성 (비전 정보 활용)
> full pick the apple and place it # 장면 분석 + TDL 생성
> quit                             # 종료
```

---

### 방법 2: 이미지 파일 사용 (Standalone)

이미 캡처된 이미지 파일을 분석합니다.

```bash
# 이미지 파일로 테스트
python interactive_pipeline_with_vision.py --mode standalone --image scene_capture.png
```

---

### 방법 3: Python 코드로 직접 사용

```python
from interactive_pipeline_with_vision import VisionEnhancedNL2TDL

# Pipeline 초기화
pipeline = VisionEnhancedNL2TDL()

# 1. 장면 분석 (이미지 파일)
scene_analysis = pipeline.analyze_scene(image_path="scene_capture.png")

# 2. TDL 생성 (비전 정보 활용)
user_command = "Pick the apple and place it in the bin"
tdl_result = pipeline.generate_tdl_with_vision(user_command, scene_analysis)

print(tdl_result['tdl_code'])
```

---

## 📋 예제 시나리오

### Scenario 1: 물체 확인 후 TDL 생성

```
> analyze

✓ Vision Analysis Complete

Scene Description:
  A robotic workspace with a table containing several objects

Detected Objects (3):
  • apple (fruit)
    Position: center of table
    Confidence: high
  • banana (fruit)
    Position: left side of table
    Confidence: high
  • milk (container)
    Position: right side of table
    Confidence: medium

> generate pick the apple

[TDL] Adding vision context to prompt...
[TDL] Generating TDL code...

✓ TDL Generation Complete

GOAL Initialize_Process()
{
    SPAWN SetTool("gripper_1") WITH WAIT;
    SPAWN SetJointVelocity(50) WITH WAIT;
}

GOAL Execute_Process()
{
    // Vision confirmed: apple is at center of table
    SPAWN MoveJ(PosJ(0, -45, 90, 0, 45, 0), 50, 100) WITH WAIT;
    SPAWN Pick("apple", 50) WITH WAIT;
    SPAWN MoveJ(PosJ(0, -30, 60, 0, 30, 0), 50, 100) WITH WAIT;
}

GOAL Finalize_Process()
{
    SPAWN MoveJ(PosJ(0, 0, 0, 0, 0, 0), 30, 50) WITH WAIT;
}
```

### Scenario 2: 존재하지 않는 물체 요청

```
> full pick the orange

[Vision] Capturing scene from MuJoCo environment...
[Vision] Analyzing image: scene_capture.png

✓ Vision Analysis Complete

Detected Objects (3):
  • apple, banana, milk

[TDL] Adding vision context to prompt...
[TDL] Generating TDL code...

⚠ Warning: 'orange' not found in detected objects.
   Available objects: apple, banana, milk

Would you like to:
1. Proceed anyway (might fail in simulation)
2. Change object name
3. Cancel
```

---

## 🔧 비전 분석 설정

### VisionSceneAnalyzer 파라미터

```python
from vision_scene_analyzer import VisionSceneAnalyzer

# 분석기 생성
analyzer = VisionSceneAnalyzer(
    api_key="YOUR_API_KEY",           # Gemini API 키
    model_name="gemini-2.0-flash-exp"  # 사용할 모델
)

# 이미지 분석
result = analyzer.analyze_scene_from_image("scene.png")

# 사용자 정의 질문
result = analyzer.analyze_scene_from_image(
    "scene.png",
    query="Is the apple close enough for the robot to reach?"
)
```

### 분석 결과 형식

```python
{
    'success': True,
    'objects': [
        {
            'name': 'apple',
            'type': 'fruit',
            'position': 'center of table',
            'confidence': 'high'
        }
    ],
    'description': 'A robotic workspace with a table...',
    'raw_response': '...'  # LLM 원본 응답
}
```

---

## 📊 장점

### vs. 비전 없는 기존 파이프라인

| 항목 | 비전 없음 | 비전 있음 |
|------|----------|----------|
| **물체 인식** | ❌ LLM이 추측 | ✅ 실제 장면 인식 |
| **위치 정보** | ❌ 없음 | ✅ 상대적 위치 제공 |
| **안전성** | 🟡 존재하지 않는 물체 명령 가능 | ✅ 실제 물체만 대상 |
| **정확도** | 🟡 일반적 TDL | ✅ 장면에 맞춤 TDL |

---

## 🎯 실제 활용 예시

### 1. 동적 물체 배치

```python
# 매번 물체 위치가 다를 때
for i in range(10):
    # 장면 변경
    randomize_object_positions(env)

    # 현재 장면 분석
    scene = pipeline.analyze_scene(env=env)

    # 자연어 명령
    tdl = pipeline.generate_tdl_with_vision(
        "Pick all fruits and sort them by type",
        scene
    )

    # TDL 실행
    execute_tdl(tdl)
```

### 2. 물체 상태 확인

```python
# 비전으로 물체 상태 확인
scene = pipeline.analyze_scene(env=env)

# 특정 물체 있는지 확인
if any(obj['name'] == 'apple' for obj in scene['objects']):
    tdl = pipeline.generate_tdl_with_vision(
        "Pick the apple",
        scene
    )
else:
    print("No apple found in scene!")
```

### 3. 멀티 스텝 태스크

```python
# Step 1: 장면 분석
scene = pipeline.analyze_scene(env=env)

# Step 2: 각 물체에 대해 TDL 생성
for obj in scene['objects']:
    if obj['type'] == 'fruit':
        tdl = pipeline.generate_tdl_with_vision(
            f"Pick the {obj['name']} and place it in the fruit bin",
            scene
        )
        execute_tdl(tdl)
```

---

## 🐛 문제 해결

### ImportError: No module named 'PIL'

**해결**: Pillow 설치
```bash
pip install Pillow
```

### Gemini Vision API 오류

**원인**: API 키 문제 또는 모델 접근 권한

**해결**:
1. `config.json`에 올바른 API 키 설정
2. Gemini Vision 모델 사용 권한 확인
3. 다른 모델 시도: `gemini-1.5-pro`, `gemini-2.0-flash-exp`

### 물체 인식 실패

**원인**: 이미지 품질 낮음 또는 물체가 작음

**해결**:
1. 이미지 해상도 높이기: `env.render(height=1080, width=1920)`
2. 카메라 각도 조정
3. 조명 개선

---

## 📖 API 레퍼런스

### VisionSceneAnalyzer

#### `analyze_scene_from_image(image_path, query=None)`
이미지 파일에서 장면 분석

**Parameters**:
- `image_path` (str): 이미지 파일 경로
- `query` (str, optional): 특정 질문

**Returns**: `dict` - 분석 결과

---

#### `analyze_scene_from_array(image_array, query=None)`
NumPy 배열에서 장면 분석

**Parameters**:
- `image_array` (np.ndarray): RGB 이미지 배열 (H, W, 3)
- `query` (str, optional): 특정 질문

**Returns**: `dict` - 분석 결과

---

#### `generate_scene_context_prompt(analysis_result)`
비전 분석 결과를 TDL 프롬프트에 추가할 컨텍스트 생성

**Parameters**:
- `analysis_result` (dict): 분석 결과

**Returns**: `str` - 프롬프트 컨텍스트

---

### VisionEnhancedNL2TDL

#### `analyze_scene(env=None, image_path=None, image_array=None)`
장면 분석 (다양한 입력 지원)

**Parameters**:
- `env` (optional): MuJoCo 환경
- `image_path` (str, optional): 이미지 파일
- `image_array` (np.ndarray, optional): 이미지 배열

**Returns**: `dict` - 분석 결과

---

#### `generate_tdl_with_vision(user_nl, scene_analysis=None)`
비전 정보를 활용한 TDL 생성

**Parameters**:
- `user_nl` (str): 사용자 자연어 명령
- `scene_analysis` (dict, optional): 비전 분석 결과

**Returns**: `dict` - TDL 생성 결과

---

## 🔬 향후 개선 방향

1. **3D 위치 추정**
   - 비전 분석 결과를 실제 3D 좌표로 변환
   - Depth 정보 활용

2. **물체 추적**
   - 연속된 프레임에서 물체 추적
   - 궤적 예측

3. **장애물 회피**
   - 비전으로 장애물 인식
   - 경로 계획에 반영

4. **실시간 피드백**
   - 태스크 실행 중 장면 모니터링
   - 오류 발생 시 자동 재계획

---

**작성일**: 2025-11-18
**버전**: 1.0.0
**상태**: ✅ 구현 완료, 테스트 대기
