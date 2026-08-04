# STEP 4 — 04. pass 키워드

**pass** = "아무 일도 안 함(통과)". 조건은 만족하지만 할 일이 없거나, 전체 골격만 잡고 내부는 나중에 채울 때 자리표시로 사용.

- 파이썬은 if 블록 안에 **최소 한 줄**이 있어야 함(비우면 IndentationError). 그 자리를 `pass`로 메꿈.
- 용도: ① 조건 맞지만 지금 할 일 없을 때 ② 뼈대 먼저 잡을 때(TODO 자리).

---

## 예제
```python
# 바구니에 딸기가 없다면 pass
basket = ['사과', '바나나']

if '딸기' not in basket:
  pass
```
**실행 결과**
```
(출력 없음)
```
`'딸기' not in basket` → 참이라 블록 진입하지만 `pass`라서 아무 동작 없이 통과.

## 자주 하는 실수
- if 블록 비워두기: 다음 줄 비우면 IndentationError → 자리표시로 `pass`.
- pass ≠ continue: pass는 아무것도 안 할 뿐. `continue`(다음 반복으로 점프)와 다름(반복문은 STEP5).
- 주석은 코드 아님: `#`만 있으면 블록이 여전히 비어 오류. 실행 가능한 자리표시는 `pass`.

## 실무 팁
강의자료엔 없지만 실무에서 "뼈대 먼저, 살은 나중" 코딩에 자주 씀(함수/조건 구조 잡고 내부는 pass). 예외처리에서 `except: pass`(오류 무시)도 있지만 진짜 오류까지 삼킬 수 있어 남용 주의.

## 퀴즈
1. pass의 역할? ① 반복 종료 ② 아무 일 안 함 ③ 오류 발생 ④ 값 반환
2. if 블록을 비우면? ① NameError ② IndentationError ③ ValueError ④ 없음
3. pass를 쓰는 상황이 아닌 것? ① 골격만 잡을 때 ② 조건 맞지만 할 일 없을 때 ③ 자리표시 ④ 반복 다음 회차로 점프
4. 코드 읽기:
```python
n = 10
if n > 5:
  pass
print("끝")
```
5. 코드 읽기:
```python
basket = ['apple', 'melon']
if 'apple' in basket:
  pass
else:
  print("없음")
```
6. 코드 작성: `items`에 `'coupon'`이 있으면 pass, 없으면 `"쿠폰 없음"` 출력.
