"""
hospital_finder.py
동물병원 위치 기반 검색 모듈
  - 공공데이터 CSV 파싱 (EPSG:5174 TM → WGS84 변환, 폐업/휴업 제거)
  - Haversine 거리 계산
  - 도보 시간 추정 (4 km/h)
  - 진단 심각도에 따른 메시지 + 가까운 병원 카드 HTML 반환
"""
import math, pandas as pd
from pathlib import Path

CSV_PATH = Path("/home/lhj/CV/data/동물_동물병원.csv")
_ACTIVE_STATUS = {'영업/정상', '영업'}
_WALK_SPEED_KMH = 4.0   # 평균 도보 속도

_transformer = None
def _get_transformer():
    global _transformer
    if _transformer is None:
        from pyproj import Transformer
        _transformer = Transformer.from_crs('EPSG:5174', 'EPSG:4326', always_xy=True)
    return _transformer


def _format_phone(raw):
    """공공데이터 전화번호(float) → 한국 표준 형식 문자열"""
    try:
        val = str(raw).strip()
        if not val or val in ('nan', 'NaN', 'None', '-', ''):
            return None
        num = '0' + str(int(float(val)))   # 앞에 0 복원
        n = len(num)
        if num.startswith('02'):
            if n == 9:   return f'{num[:2]}-{num[2:5]}-{num[5:]}'    # 02-XXX-XXXX
            if n == 10:  return f'{num[:2]}-{num[2:6]}-{num[6:]}'    # 02-XXXX-XXXX
        elif num.startswith(('010','011','016','017','018','019')):
            if n == 11:  return f'{num[:3]}-{num[3:7]}-{num[7:]}'    # 01X-XXXX-XXXX
        else:
            if n == 10:  return f'{num[:3]}-{num[3:6]}-{num[6:]}'    # 0XX-XXX-XXXX
            if n == 11:  return f'{num[:3]}-{num[3:7]}-{num[7:]}'    # 0XX-XXXX-XXXX
        return num  # 비표준 자릿수 → 그대로 표시
    except Exception:
        return None


def _walk_time_str(dist_km):
    """거리(km) → 도보 소요 시간 문자열"""
    minutes = dist_km / _WALK_SPEED_KMH * 60
    if minutes < 1:
        return '도보 1분 미만'
    return f'도보 약 {round(minutes)}분'


def _car_time_str(dist_km):
    """거리(km) → 차량 소요 시간 문자열 (평균 30km/h 시내 주행)"""
    minutes = dist_km / 30.0 * 60
    if minutes < 1:
        return '차량 1분 미만'
    return f'차량 약 {round(minutes)}분'


def load_hospitals():
    """CSV 로드 → 영업중만 필터 → WGS84 변환 → DataFrame 반환"""
    if not CSV_PATH.exists():
        return None

    for enc in ['cp949', 'utf-8-sig', 'euc-kr', 'utf-8']:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc, low_memory=False)
            break
        except Exception:
            continue
    else:
        return None

    if '영업상태명' in df.columns:
        df = df[df['영업상태명'].isin(_ACTIVE_STATUS)]

    xcol, ycol = '좌표정보(X)', '좌표정보(Y)'
    if xcol not in df.columns or ycol not in df.columns:
        return None

    df = df.dropna(subset=[xcol, ycol]).copy()
    df[xcol] = pd.to_numeric(df[xcol], errors='coerce')
    df[ycol] = pd.to_numeric(df[ycol], errors='coerce')
    df = df.dropna(subset=[xcol, ycol])
    df = df[(df[xcol].between(100000, 600000)) & (df[ycol].between(300000, 600000))]

    t = _get_transformer()
    lons, lats = t.transform(df[xcol].values, df[ycol].values)
    df = df.copy()
    df['_lat'] = lats
    df['_lon'] = lons
    df = df[(df['_lat'].between(33, 38)) & (df['_lon'].between(125, 130))]
    df = df.reset_index(drop=True)
    return df


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d = math.pi / 180
    a = (math.sin((lat2-lat1)*d/2)**2 +
         math.cos(lat1*d) * math.cos(lat2*d) * math.sin((lon2-lon1)*d/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def find_nearby(lat, lon, n=5):
    """사용자 위치 기준 가까운 병원 n개 반환"""
    df = load_hospitals()
    if df is None or len(df) == 0:
        return []

    dists = [haversine(lat, lon, la, lo)
             for la, lo in zip(df['_lat'].values, df['_lon'].values)]
    df = df.copy()
    df['_dist_km'] = dists
    df = df.nsmallest(n, '_dist_km')

    addr_col = '도로명주소' if '도로명주소' in df.columns else '지번주소'
    hospitals = []
    for _, row in df.iterrows():
        phone = _format_phone(row.get('전화번호'))
        hospitals.append({
            'name':  str(row['사업장명']) if '사업장명' in df.columns else '이름 미상',
            'addr':  str(row[addr_col])  if addr_col  in df.columns else '주소 미상',
            'phone': phone,
            'dist':  row['_dist_km'],
        })
    return hospitals


_URGENCY_MAP = {0: 1, 1: 2, 2: 2, 3: 2, 4: 0}  # 5-class: 태선화·농포·궤양·결절·정상

def h_hospital_section(dis_id, hospitals):
    """진단 결과 + 주변 병원 HTML"""
    if not hospitals:
        return ''

    urg = _URGENCY_MAP.get(dis_id, 0)
    if urg == 2:
        banner_col = '#F44336'
        banner_txt = '🚨 즉시 동물병원 방문이 필요합니다. 가까운 병원을 확인하세요.'
    elif urg == 1:
        banner_col = '#FF9800'
        banner_txt = '⚠️ 동물병원 방문을 권장합니다. 가까운 병원을 확인하세요.'
    else:
        banner_col = '#4CAF50'
        banner_txt = '✅ 정상 범위입니다. 정기 검진을 위해 근처 병원을 참고하세요.'

    banner = (
        f'<div style="background:{banner_col}22;border-left:4px solid {banner_col};'
        f'border-radius:6px;padding:10px 14px;color:{banner_col};font-weight:bold;'
        f'font-size:13px;margin-bottom:12px;">{banner_txt}</div>'
    )

    cards = ''
    for h in hospitals:
        dist_km   = h['dist']
        dist_str  = f'{dist_km*1000:.0f}m' if dist_km < 1 else f'{dist_km:.1f}km'
        walk_str  = _walk_time_str(dist_km)
        car_str   = _car_time_str(dist_km)
        phone_str = h['phone'] if h['phone'] else '번호 미상'
        cards += (
            f'<div style="background:#ffffff;border-radius:8px;padding:12px 14px;margin-bottom:8px;border:1px solid #e2e8f0;">'
            # 병원명 + 거리/도보/차량 배지
            f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;">'
            f'<div style="color:#1a2a3a;font-weight:bold;font-size:13px;">🏥 {h["name"]}</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;">'
            f'<span style="background:#eef2ff;color:#2563eb;font-size:11px;padding:2px 8px;border-radius:10px;">'
            f'📏 {dist_str}</span>'
            f'<span style="background:#d1fae5;color:#166534;font-size:11px;padding:2px 8px;border-radius:10px;">'
            f'🚶 {walk_str}</span>'
            f'<span style="background:#fef3c7;color:#92400e;font-size:11px;padding:2px 8px;border-radius:10px;">'
            f'🚗 {car_str}</span>'
            f'</div></div>'
            # 주소
            f'<div style="color:#6b7280;font-size:11px;margin-top:6px;">📍 {h["addr"]}</div>'
            # 전화번호
            f'<div style="color:#7db8f0;font-size:12px;margin-top:4px;font-weight:bold;">📞 {phone_str}</div>'
            f'</div>'
        )

    return (
        f'<div style="background:#f8fafc;border-radius:8px;padding:14px;margin-top:10px;border:1px solid #e2e8f0;">'
        f'<div style="color:#1a2a3a;font-weight:bold;font-size:14px;margin-bottom:10px;">'
        f'📍 가까운 동물병원</div>'
        f'{banner}{cards}'
        f'<div style="color:#6b7280;font-size:10px;margin-top:6px;text-align:right;">'
        f'공공데이터 기반 | 실제 영업 여부 방문 전 확인 권장</div>'
        f'</div>'
    )
