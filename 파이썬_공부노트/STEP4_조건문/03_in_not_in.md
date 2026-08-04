# STEP 4 — 03. in / not in 연산자

**in / not in** = 데이터 안에 찾는 것이 **있는지/없는지** 확인하는 연산자. 결과는 항상 **참(True)/거짓(False)**.

- `A in B` → B 안에 A가 **있으면** True
- `A not in B` → B 안에 A가 **없으면** True (in의 반대)

문자열(글자 포함?), 리스트(요소 포함?), 딕셔너리(키 있음?)에 사용. 조건문(if)과 짝으로 많이 씀.

---

## 예제
```python
# 'python' 문자열에 k가 없으면 True, 있으면 False 출력
if 'k' not in 'python':
    print(True)
else:
    print(False)
```
**실행 결과**
```
True
```
`'python'` 안에 `'k'` 없음 → `not in`이 참 → `print(True)` 실행.

## 자주 하는 실수
- 딕셔너리에서 `key in dict`는 **키**를 검사. 값은 `in dict.values()`.
- 대소문자 구분: `'P' in 'python'` → False.
- in/not in 반대로 예상: `'a' in 'banana'` → True, `'a' not in 'banana'` → False.

## 실무 팁
강의자료엔 없지만 실무에서 자주: `if user_id in banned_list:` (차단 검사), `if 'email' in form_data:` (키 존재 확인), `if country in ['한국','일본','중국']:` (여러 값 중 하나인지 한 줄 검사).

## 퀴즈
1. in의 결과 종류? ① 숫자 ② 문자열 ③ 부울 ④ 리스트
2. `'k' not in 'python'`? ① True ② False ③ 오류 ④ None
3. 딕셔너리 `key in dict`는? ① 값 ② 키 ③ 길이 ④ 타입
4. 코드 읽기:
```python
fruits = ['apple', 'banana']
if 'melon' in fruits:
  print("있음")
else:
  print("없음")
```
5. 코드 읽기:
```python
print('a' in 'banana')
print('z' not in 'banana')
```
6. 코드 작성: `basket = ['사과','바나나']`에 `'포도'`가 없으면 `"포도 없음"` 출력(not in).
