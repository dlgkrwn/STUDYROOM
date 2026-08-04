# STEP 4 — 02. else · elif 문 (조건 분기)

- **else 문**: if 조건이 **거짓일 때** 실행되는 부분. "아니면 이걸 해라".
- **elif 문**: else + if. **세 개 이상의 조건**을 연결할 때 사용.

## ⭐ 구조
```
if 조건1:      # 조건1 참이면 여기
elif 조건2:    # 아니고 조건2 참이면 여기
elif 조건3:    # 아니고 조건3 참이면 여기
else:          # 위가 전부 거짓이면 여기
```
- `if` 맨 위 1개 / `elif` 여러 개 가능 / `else` 맨 아래 1개(생략 가능)
- 위에서부터 검사 → **처음 참이 된 블록 하나만** 실행하고 종료.
- else·elif도 **콜론 `:` + 들여쓰기** 필수.

---

## 1. else 문 — 거짓일 때
```python
selena_lecture = True

if selena_lecture:
  print("파이썬을 정복할 수 있다.")
else :
  print("파이썬을 정복할 수 없다.")
```
**실행 결과**
```
파이썬을 정복할 수 있다.
```
`selena_lecture`가 True → if만 실행, else는 건너뜀. (False였다면 else 실행.)

## 2. elif 문 — 여러 갈래
```python
score = "C"

if score == "A" :
  print("A class")
elif score == "B" :
  print("B class")
elif score == "C" :
  print("C class")
else :
  print("D class")
```
**실행 결과**
```
C class
```
위에서부터 검사 → `score == "C"`가 처음으로 참 → `"C class"`만 출력. (`==`는 '같다' 비교, 대입 `=`와 구분.)

## 자주 하는 실수
- elif 대신 if 여러 개 → 각각 독립 검사가 되어 여러 블록이 실행될 수 있음. "하나만" 고르려면 elif.
- else에 조건 붙이기: `else x == 1:` ❌ → else는 조건 없이 `else:`.
- 콜론/들여쓰기 누락(else·elif도 동일).
- 순서 실수: 넓은 조건을 위에 두면 아래 조건까지 도달 못 함. 좁은 → 넓은 순서 주의.

## 실무 팁
강의자료엔 없지만 실무에서는 "점수 → 등급", "나이 → 연령대"처럼 **구간 나누기**에 if–elif–else를 자주 씀. elif가 너무 길어지면 딕셔너리/구간 함수로 리팩터링하기도 함.

## 퀴즈
1. else 문은? ① 참일 때 실행 ② if가 거짓일 때 실행 ③ 항상 실행 ④ 조건 필요
2. 3개 이상 조건 연결? ① else ② elif ③ end ④ and
3. if–elif–else에서 실행되는 블록 수? ① 모두 ② 참인 것 전부 ③ 처음 참 하나 ④ 없음
4. 코드 읽기:
```python
temp = 15
if temp >= 28:
  print("더움")
elif temp >= 10:
  print("적당")
else:
  print("추움")
```
5. 코드 읽기:
```python
x = 5
if x > 10:
  print("A")
elif x > 3:
  print("B")
elif x > 1:
  print("C")
```
6. 코드 작성: `age=20`일 때 `age >= 19`면 `"성인"`, 아니면 `"미성년자"` 출력(if–else).
