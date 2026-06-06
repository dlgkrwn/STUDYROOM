"""
06_app.py - 반려동물 피부 질환 진단 시스템
Pipeline: YOLO26m → SAM2.1 → SigLIP 2 SO400M (2025, 5-class)
분류 대상: 태선화 · 농포/여드름 · 미란/궤양 · 결절/종괴 · 정상
※ 구진/플라크·인설/가피는 시각적 모호성으로 분류 범위에서 제외
"""
import os, numpy as np, cv2, torch, torch.nn as nn, gradio as gr
from transformers import AutoProcessor, AutoModel
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from pathlib import Path
from datetime import datetime
import sys; sys.path.insert(0, str(Path(__file__).parent))
from hospital_finder import find_nearby, h_hospital_section

_kr_font = Path('/home/lhj/.local/share/fonts/malgun.ttf')
if _kr_font.exists():
    _fm.fontManager.addfont(str(_kr_font))
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = Path("/home/lhj/CV/data")
NORMAL_CONF_THRESH = 0.65   # Normal 예측 최소 확신도 (미달 시 의심 질환 표시)
YOLO_W        = DATA_DIR / "results/yolo/yolo26m_det_v2/weights/best.pt"
SIGLIP2_FT_W  = DATA_DIR / "weights/siglip2_6class.pt"
SIGLIP2_ID    = "google/siglip2-so400m-patch14-384"
SAM2_W        = DATA_DIR / "weights/sam2.1_hiera_tiny.pt"
NUM_CLASSES   = 5
NORMAL_IDX    = 4   # 정상 클래스 인덱스 (5-class)

# 8분류 신체 부위 (v2 모델용), 4분류 fallback
REGION8_KR = ['얼굴·귀', '목', '등', '옆구리·가슴',
               '앞발·앞다리', '뒷발·뒷다리', '복부', '기타']
REGION4_KR = ['머리 (얼굴·귀)', '몸통 (등·옆구리)', '사지 (다리·발)', '복부 (배)']
REGION_KR  = REGION8_KR  # 앱 표시용 기본

# ── 질환별 상세 정보 (5-class) ──
DISEASE_NAMES_KR    = ['태선화', '농포/여드름', '미란/궤양', '결절/종괴', '정상']
DISEASE_COLORS_LIST = ['#FF5722', '#9C27B0', '#F44336', '#607D8B', '#4CAF50']

DISEASE_INFO = {
    0: {
        'name':   '태선화',
        'icon':   '🟠',
        'eng':    'Lichenification',
        'desc':   '피부가 두꺼워지고 주름이 두드러지는 만성 변화입니다. 지속적인 긁기·핥기로 인해 발생합니다.',
        'cause':  '만성 알레르기, 반복적 자극, 아토피 피부염',
        'action': '만성 질환 가능성 높음 — 수의사 정밀 진료 권장',
        'urgency': 1,
        'color':  '#FF5722',
    },
    1: {
        'name':   '농포/여드름',
        'icon':   '🟣',
        'eng':    'Pustule / Acne',
        'desc':   '고름이 찬 작은 종기가 피부에 생긴 상태입니다. 세균 감염이 주요 원인입니다.',
        'cause':  '세균성 피부염(농피증), 모낭 감염',
        'action': '항생제 치료 필요 — 빠른 동물병원 방문 권장',
        'urgency': 2,
        'color':  '#9C27B0',
    },
    2: {
        'name':   '미란/궤양',
        'icon':   '🔴',
        'eng':    'Erosion / Ulcer',
        'desc':   '피부 표면이 손상되어 짓무르거나 패인 상처가 생긴 상태입니다. 출혈이나 삼출물이 동반될 수 있습니다.',
        'cause':  '외상, 심한 감염, 자가면역 질환',
        'action': '🚨 즉시 동물병원 방문 필요',
        'urgency': 2,
        'color':  '#F44336',
    },
    3: {
        'name':   '결절/종괴',
        'icon':   '⚫',
        'eng':    'Nodule / Mass',
        'desc':   '피부 아래 딱딱하거나 부드러운 덩어리가 만져지는 상태입니다. 종양 가능성이 있습니다.',
        'cause':  '낭종, 지방종, 종양 (양성·악성 포함)',
        'action': '🚨 조직 검사 등 정밀 검사 즉시 필요',
        'urgency': 2,
        'color':  '#607D8B',
    },
    4: {
        'name':   '정상',
        'icon':   '✅',
        'eng':    'Normal',
        'desc':   '현재 피부 상태는 정상 범위입니다. 특별한 이상 소견이 발견되지 않았습니다.',
        'cause':  '—',
        'action': '정기적인 피부 관찰 권장 (1개월 주기)',
        'urgency': 0,
        'color':  '#4CAF50',
    },
}

DISEASE_SLIDERS = {
    0: [10, 30, 45, 45],   # 태선화
    1: [20, 65, 10, 40],   # 농포/여드름
    2: [65, 25, 10, 30],   # 미란/궤양
    3: [20, 35, 20, 35],   # 결절/종괴
    4: [0,   5,  5, 85],   # 정상
}

DISEASE_STEPS = {
    0: {
        'step1': '엘리자베스 칼라 착용 → 핥기·긁기 즉시 차단',
        'step2': '증상 기간·악화 패턴 기록 / 알레르기 이력 준비',
        'step3': '스트레스 요인 제거 / 환경 청결 유지 / 보습 관리',
        'step4': ['피부 두꺼워짐이 급격히 진행되는 경우', '2차 감염(고름·냄새) 발생 시'],
    },
    1: {
        'step1': '병변 짜거나 터뜨리기 절대 금지 / 칼라 착용',
        'step2': '항생제 치료 이력 준비 / 병변 사진 촬영',
        'step3': '해당 부위 청결 유지 / 타 반려동물 접촉 제한',
        'step4': ['발열·무기력 동반 시 즉시 내원', '병변이 전신으로 번지는 경우'],
    },
    2: {
        'step1': '🚨 즉시 동물병원 방문 / 자가 소독·연고 금지',
        'step2': '엘리자베스 칼라 착용 / 상처 발생 원인·시기 기록',
        'step3': '청결하고 건조한 환경 유지 / 수의사 처방 후 처치',
        'step4': ['출혈이 멈추지 않는 경우', '상처 주변이 검게 변하거나 냄새 심한 경우'],
    },
    3: {
        'step1': '덩어리 강하게 누르거나 터뜨리기 절대 금지',
        'step2': '발견 시기·크기 변화 사진으로 기록 / 전신 추가 덩어리 확인',
        'step3': '주변 자극 최소화 / 정기 크기 모니터링',
        'step4': ['빠른 크기 증가', '색 변화·출혈·궤양화 발생 시'],
    },
    4: {
        'step1': '현재 특별한 조치 불필요 — 정상 상태 유지',
        'step2': '정기 피부 검진 일정 확인 (3~6개월)',
        'step3': '규칙적인 브러싱·목욕 / 균형 잡힌 영양 공급',
        'step4': ['새로운 병변 발생 시 재촬영 권장', '가려움·탈모·붉어짐 관찰'],
    },
}

_state = {'dis_id': 4, 'reg_id': 1, 'pet_name': '반려동물', 'probs': None, 'image': None}

# ── AI 소견 템플릿 (5-class) ──
_OPINION_MAIN = {
    0: {
        'high':   "피부 태선화 소견이 뚜렷하게 확인됩니다. 피부가 두꺼워지고 주름이 현저하게 두드러지며, 색소침착도 동반됩니다. 만성적인 긁기·핥기 자극 또는 오래된 알레르기 반응에 의한 전형적인 변화입니다.",
        'mid':    "피부 태선화 초기 변화가 관찰됩니다. 피부 두께 증가 및 주름 강조가 보이며, 만성 자극이 지속될 경우 악화될 수 있습니다.",
        'low':    "태선화 의심 소견이 있으나 초기 변화와 감별이 필요합니다. 긁기·핥기 행동 여부를 추가로 확인하시기 바랍니다.",
    },
    1: {
        'high':   "피부에 농포가 명확하게 관찰됩니다. 고름이 찬 작은 종기와 발적이 확인되며, 세균성 피부염(농피증)에 전형적인 소견입니다. 항생제 치료가 필요할 가능성이 높습니다.",
        'mid':    "농포/여드름 소견이 의심됩니다. 소규모 농성 병변이 확인되며, 2차 세균 감염 여부를 평가해야 합니다.",
        'low':    "농포 의심 소견이 있으나 초기 단계로 보입니다. 병변 부위를 청결하게 유지하고 경과를 관찰하시기 바랍니다.",
    },
    2: {
        'high':   "피부 미란 및 궤양 소견이 확인됩니다. 피부 표면이 손상되어 짓무르거나 패인 상처가 관찰되며, 출혈 또는 삼출물 동반 가능성이 있습니다. 즉각적인 처치가 필요한 상태입니다.",
        'mid':    "피부 미란/궤양 초기 소견이 관찰됩니다. 표피 손상이 진행 중이며, 2차 감염 예방을 위한 빠른 조치가 필요합니다.",
        'low':    "미란/궤양 의심 소견이 있습니다. 상처 여부를 직접 확인하고 이상 소견 시 즉시 내원하시기 바랍니다.",
    },
    3: {
        'high':   "피부 내 결절/종괴 소견이 명확히 확인됩니다. 경계가 있는 덩어리가 관찰되며, 낭종·지방종·종양(양성·악성 포함)의 가능성이 있습니다. 조직 검사 등 정밀 평가가 필요합니다.",
        'mid':    "피부 결절/종괴 소견이 의심됩니다. 피부 융기 또는 피하 종괴가 관찰되며, 크기 변화 추적 관찰이 필요합니다.",
        'low':    "결절/종괴 의심 소견이 있으나 확인이 어렵습니다. 촉진 시 덩어리 확인 여부와 함께 수의사 평가를 권장합니다.",
    },
    4: {
        'high':   "현재 관찰되는 피부 상태는 정상 범위에 해당합니다. 병변·발적·각질·종창 등 이상 소견이 뚜렷하게 확인되지 않으며, 피부 상태는 전반적으로 양호합니다.",
        'mid':    "피부 상태는 대체로 정상으로 평가됩니다. 일부 미세한 변화가 있으나 임상적으로 유의한 수준은 아닙니다. 정기적인 관찰을 권장합니다.",
        'low':    "정상으로 분류되었으나 확신도가 낮습니다. 다른 각도에서 재촬영 후 재평가를 권장합니다.",
    },
}

_OPINION_ACTION = {
    0: {'high': '엘리자베스 칼라로 즉시 자극을 차단하고, 수의사 정밀 진료를 받으세요.',
        'mid':  '긁기·핥기 행동을 억제하고 알레르기 원인 파악을 시작하세요.',
        'low':  '환경 스트레스 요인을 점검하고 경과를 관찰하세요.'},
    1: {'high': '🚨 병변을 짜거나 터뜨리지 마세요. 빠른 동물병원 방문이 필요합니다.',
        'mid':  '해당 부위 청결 유지 후 빠른 시일 내 수의사 진료를 받으세요.',
        'low':  '청결 유지 관찰 후, 악화 시 즉시 내원하세요.'},
    2: {'high': '🚨 즉시 동물병원 방문 필요. 자가 소독·연고 적용을 금지하세요.',
        'mid':  '🚨 하루 이내 동물병원 방문을 권장합니다.',
        'low':  '상처 상태를 즉시 확인하고, 이상 시 당일 내원하세요.'},
    3: {'high': '🚨 즉시 정밀 검사(초음파·조직검사) 필요. 덩어리를 누르거나 터뜨리지 마세요.',
        'mid':  '빠른 시일 내 수의사 진료를 받고 종괴 크기를 기록하세요.',
        'low':  '크기 변화를 주기적으로 사진으로 기록하고 수의사와 상담하세요.'},
    4: {'high': '정기 피부 검진(3~6개월 주기) 및 균형 잡힌 영양 공급을 유지하세요.',
        'mid':  '정기 관찰을 유지하고 이상 소견 발생 시 재촬영하세요.',
        'low':  '더 명확한 사진으로 재촬영 후 재평가를 권장합니다.'},
}


def generate_ai_opinion(image_rgb, dis_id, probs):
    """질환 분류 결과를 기반으로 임상 소견 텍스트 생성"""
    conf = float(probs[dis_id]) * 100
    tier = 'high' if conf >= 70 else ('mid' if conf >= 50 else 'low')

    main_text   = _OPINION_MAIN[dis_id][tier]
    action_text = _OPINION_ACTION[dis_id][tier]

    # 감별 진단: 2위 항목이 20% 이상이면 언급
    sorted_ids = np.argsort(probs)[::-1]
    ddx_id  = int(sorted_ids[1])
    ddx_conf = float(probs[ddx_id]) * 100
    ddx_html = ''
    if ddx_conf >= 20 and ddx_id != NORMAL_IDX:
        ddx_html = (
            f'<div style="background:#f8fafc;border-left:3px solid #f0c040;border-radius:0 4px 4px 0;'
            f'padding:6px 10px;margin-top:8px;font-size:12px;color:#1e293b;">'
            f'🔍 <b style="color:#f0c040;">감별 진단 고려:</b> '
            f'{DISEASE_NAMES_KR[ddx_id]} ({ddx_conf:.0f}%) — 유사 병변과의 감별이 필요합니다.'
            f'</div>'
        )

    conf_color = '#4CAF50' if conf >= 70 else ('#FF9800' if conf >= 50 else '#F44336')
    return main_text, action_text, ddx_html, conf_color, f'{conf:.0f}%'


def h_ai_opinion(main_text='', action_text='', ddx_html='', conf_color='#555', conf_pct=''):
    if not main_text:
        return ('<div style="background:#f8fafc;border-radius:8px;padding:14px;'
                'text-align:center;color:#1e293b;font-size:12px;">분석 후 표시됩니다</div>')
    return (
        f'<div style="background:#f8fafc;border-radius:8px;padding:14px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">'
        f'<div style="color:#1a2a3a;font-weight:bold;font-size:14px;">🤖 AI 소견란</div>'
        f'<span style="background:{conf_color};color:white;font-size:11px;padding:2px 10px;'
        f'border-radius:12px;">확신도 {conf_pct}</span></div>'
        f'<div style="color:#1e293b;font-size:13px;line-height:1.9;background:#ffffff;'
        f'border-radius:6px;padding:12px 14px;margin-bottom:8px;">{main_text}</div>'
        f'<div style="background:#eef2ff;border-radius:6px;padding:9px 12px;font-size:12px;">'
        f'<b style="color:#1d4ed8;">📋 권장 조치</b><br>'
        f'<span style="color:#1e293b;">{action_text}</span></div>'
        f'{ddx_html}'
        f'<div style="color:#374151;font-size:10px;margin-top:8px;text-align:right;">'
        f'⚠️ AI 보조 소견 — 수의사의 최종 진단을 반드시 받으시기 바랍니다</div>'
        f'</div>'
    )


# ── Score ──
# 긴급도별 기준 점수 (urgency 0=정상 / 1=주의 / 2=긴급)
_URG_BASE = {0: 15, 1: 47, 2: 80}

def disease_score(dis_id, probs):
    """진단 긴급도 + 확신도 → 게이지 점수 (0~100)
    urgency 구간을 항상 유지하면서 확신도로 ±8점 조정"""
    urg  = DISEASE_INFO[dis_id]['urgency']
    base = _URG_BASE[urg]
    conf = float(probs[dis_id])
    adj  = (conf - 0.5) * 16          # 확신도 50% 기준 ±8점
    return float(min(100, max(0, base + adj)))

def severity_label(s):
    if s < 33:   return "정상 범위",              "#4CAF50"
    elif s < 66: return "지속적 관찰 요망 (주의)", "#FF9800"
    else:        return "즉시 진료 필요 (위험)",   "#F44336"


# ── Gauge ──
def make_gauge(score):
    fig = plt.figure(figsize=(5, 3.8), facecolor='#0d1b2a')
    ax  = fig.add_subplot(111, facecolor='#0d1b2a')
    for (s1,s2),c in [((0,33),'#2ecc71'),((33,66),'#f39c12'),((66,100),'#e74c3c')]:
        a1=np.pi*(1-s1/100); a2=np.pi*(1-s2/100)
        th=np.linspace(a1,a2,80)
        xo,yo=np.cos(th),np.sin(th)
        xi,yi=np.cos(th[::-1])*.65,np.sin(th[::-1])*.65
        ax.fill(np.r_[xo,xi],np.r_[yo,yi],color=c,alpha=0.85)
    ang=np.pi*(1-score/100)
    ax.annotate('',xy=(np.cos(ang)*.88,np.sin(ang)*.88),xytext=(0,0),
                arrowprops=dict(arrowstyle='->',color='white',lw=2.5))
    ax.plot(0,0,'o',color='white',ms=7,zorder=5)
    lbl,col=severity_label(score)
    ax.text(0,0.28,f'{score:.1f}',ha='center',fontsize=34,fontweight='bold',color=col)
    ax.text(0,0.06,'SEVERITY SCORE',ha='center',fontsize=8,color='#aaa')
    for xp,t in [(-1.15,'0'),(0,'50'),(1.15,'100')]:
        ax.text(xp,0.06,t,ha='center',fontsize=8,color='#777')
    lo=0 if score<33 else (33 if score<66 else 66)
    hi=33 if score<33 else (66 if score<66 else 100)
    ax.text(0,-0.18,f'중증도 구간 ({lo}-{hi})',ha='center',fontsize=8,color='#aaa',
            bbox=dict(boxstyle='round,pad=0.3',fc='#1a2a3a',alpha=0.9))
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.35,1.3); ax.axis('off')
    plt.tight_layout(pad=0.3)
    return fig


# ── HTML 빌더 ──
def h_prob_bars(probs):
    """AI 분류 확률 상위 5개 막대 차트"""
    if probs is None: return ''
    order = np.argsort(probs)[::-1][:5]
    bars = ''
    for i in order:
        pct = float(probs[i]) * 100
        col = DISEASE_COLORS_LIST[i]
        bars += (
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
            f'<div style="width:84px;font-size:11px;color:#64748b;text-align:right;white-space:nowrap;">'
            f'{DISEASE_NAMES_KR[i]}</div>'
            f'<div style="flex:1;background:#e2e8f0;border-radius:3px;height:12px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;background:{col};height:100%;border-radius:3px;"></div></div>'
            f'<div style="width:38px;font-size:11px;color:{col};font-weight:bold;">{pct:.1f}%</div>'
            f'</div>'
        )
    return (
        f'<div style="background:#ffffff;border-radius:8px;padding:10px 14px;margin-top:10px;">'
        f'<div style="color:#1e293b;font-size:11px;margin-bottom:6px;">📊 AI 분류 확률 (상위 5개)</div>'
        f'{bars}</div>'
    )


def h_disease_card(dis_id, conf_pct='', probs=None):
    info = DISEASE_INFO[dis_id]
    urg  = info['urgency']
    bg   = '#d1fae5' if urg==0 else ('#fef3c7' if urg==1 else '#fee2e2')
    badge_col = '#4CAF50' if urg==0 else ('#FF9800' if urg==1 else '#F44336')
    badge_txt = '정상' if urg==0 else ('주의' if urg==1 else '긴급')
    conf_html = f'<span style="font-size:11px;color:#1e293b;margin-left:8px;">{conf_pct}</span>' if conf_pct else ''
    main_html = (
        f'<div style="background:{bg};border-left:5px solid {info["color"]};'
        f'border-radius:10px;padding:16px 20px;margin:6px 0;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">'
        f'<div style="font-size:28px;font-weight:bold;color:#1a2a3a;">'
        f'{info["icon"]} {info["name"]}'
        f'<span style="font-size:13px;color:#1e293b;font-weight:normal;margin-left:10px;">{info["eng"]}</span>'
        f'{conf_html}</div>'
        f'<span style="background:{badge_col};color:white;padding:3px 12px;border-radius:20px;'
        f'font-size:12px;font-weight:bold;">{badge_txt}</span></div>'
        f'<div style="color:#1e293b;font-size:13px;line-height:1.8;margin-bottom:8px;">'
        f'📋 {info["desc"]}</div>'
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
        f'<div style="background:#f8fafc;border-radius:6px;padding:8px 14px;font-size:12px;color:#1e293b;">'
        f'🦠 <b style="color:#64748b;">주요 원인</b><br>{info["cause"]}</div>'
        f'<div style="background:#f8fafc;border-radius:6px;padding:8px 14px;font-size:12px;">'
        f'<b style="color:{badge_col};">권장 조치</b><br>'
        f'<span style="color:#1e293b;">{info["action"]}</span></div>'
        f'</div>'
        f'{h_prob_bars(probs)}'
        f'</div>'
    )
    return main_html

def h_stats(final, erythema_c, scale_c, ulcer_c, probs=None):
    # 1행: 중증도 점수 + 3개 기여값
    score_cards = [('최종 점수', f'{final:.1f}점', '#f0c040'),
                   ('홍반 기여',  f'{erythema_c:.1f}', '#e74c3c'),
                   ('각질 기여',  f'{scale_c:.1f}',    '#3498db'),
                   ('궤양 기여',  f'{ulcer_c:.1f}',    '#9b59b6')]
    row1 = ''.join(
        f'<div style="background:#eef2ff;border-radius:10px;padding:10px 18px;min-width:90px;">'
        f'<div style="color:#1e293b;font-size:11px;">{l}</div>'
        f'<div style="color:{c};font-size:22px;font-weight:bold;">{v}</div></div>'
        for l,v,c in score_cards)
    # 2행: 6개 질환별 AI 예측 확률
    row2 = ''
    if probs is not None:
        for i in range(NUM_CLASSES):
            pct = float(probs[i]) * 100
            col = DISEASE_COLORS_LIST[i]
            row2 += (
                f'<div style="background:#eef2ff;border-radius:8px;padding:8px 12px;min-width:82px;'
                f'border-top:2px solid {col};">'
                f'<div style="color:#1e293b;font-size:10px;white-space:nowrap;margin-bottom:2px;">'
                f'{DISEASE_NAMES_KR[i]}</div>'
                f'<div style="color:{col};font-size:18px;font-weight:bold;">{pct:.1f}%</div>'
                f'</div>'
            )
        row2 = (f'<div style="color:#64748b;font-size:10px;margin:6px 0 3px;">🧬 질환별 AI 예측</div>'
                f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{row2}</div>')
    return (f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{row1}</div>{row2}')

def h_alert(score):
    lbl,col = severity_label(score)
    bg  = "#d1fae5" if score<33 else ("#fef3c7" if score<66 else "#fee2e2")
    icon= "✅" if score<33 else ("⚠️" if score<66 else "🚨")
    return (f'<div style="background:{bg};border-left:4px solid {col};border-radius:6px;'
            f'padding:10px 16px;color:{col};font-weight:bold;font-size:14px;margin:4px 0;">'
            f'{icon}&nbsp; {lbl}</div>')

def h_patient(dis_name, reg_name, tta_info=''):
    now  = datetime.now().strftime('%Y-%m-%d %H:%M')
    name = _state['pet_name']
    tta  = f'<br><span style="color:#1d4ed8;font-weight:bold;">🔬 분석 방식</span>&nbsp;&nbsp;<span style="color:#1e293b;">{tta_info}</span>' if tta_info else ''
    lbl  = 'color:#2563eb;font-weight:bold;'
    val  = 'color:#1e293b;'
    return (f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;'
            f'font-size:13px;margin-top:6px;line-height:2.2;">'
            f'<span style="{lbl}">🐾 환자</span>&nbsp;&nbsp;<span style="{val}">{name}</span><br>'
            f'<span style="{lbl}">📅 분석 날짜</span>&nbsp;&nbsp;<span style="{val}">{now}</span><br>'
            f'<span style="{lbl}">📍 질환 부위</span>&nbsp;&nbsp;<span style="{val}">{reg_name}</span><br>'
            f'<span style="{lbl}">🔬 진단 결과</span>&nbsp;&nbsp;<span style="{val}">{dis_name}</span>'
            f'{tta}</div>')

def h_guidance(score, dis_id):
    info  = DISEASE_INFO[dis_id]
    steps = DISEASE_STEPS[dis_id]
    now   = datetime.now().strftime('%Y-%m-%d')
    if score < 33:
        schedule = "1주일 후 재촬영 권장"
        sched_col = '#4CAF50'
    elif score < 66:
        schedule = "3일 후 재점검"
        sched_col = '#FF9800'
    else:
        schedule = "오늘 중 동물병원 방문"
        sched_col = '#F44336'
    warn_items = ''.join(
        f'<li style="color:#dc2626;margin:2px 0;">{w}</li>'
        for w in steps['step4'])
    return (
        f'<div style="background:#f8fafc;border-radius:8px;padding:14px;color:#64748b;">'
        f'<div style="color:#1a2a3a;font-weight:bold;font-size:14px;margin-bottom:10px;">🛡️ AI 1차 가이드</div>'
        # 진단 요약
        f'<div style="background:#eef2ff;border-radius:6px;padding:10px;font-size:12px;'
        f'line-height:1.8;margin-bottom:8px;display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:26px;">{info["icon"]}</span>'
        f'<div><b style="color:{info["color"]};font-size:13px;">{info["name"]}</b>'
        f'<div style="color:#1e293b;font-size:11px;">{info["desc"]}</div></div></div>'
        # STEP 1
        f'<div style="margin-bottom:6px;">'
        f'<div style="color:#1d4ed8;font-size:11px;font-weight:bold;margin-bottom:3px;">① 즉시 조치</div>'
        f'<div style="background:#f0f4ff;border-left:3px solid #7db8f0;border-radius:0 4px 4px 0;'
        f'padding:7px 10px;font-size:12px;color:#1e293b;">{steps["step1"]}</div></div>'
        # STEP 2
        f'<div style="margin-bottom:6px;">'
        f'<div style="color:#f0c040;font-size:11px;font-weight:bold;margin-bottom:3px;">② 병원 방문 전 준비</div>'
        f'<div style="background:#f0f4ff;border-left:3px solid #f0c040;border-radius:0 4px 4px 0;'
        f'padding:7px 10px;font-size:12px;color:#1e293b;">{steps["step2"]}</div></div>'
        # STEP 3
        f'<div style="margin-bottom:6px;">'
        f'<div style="color:#2ecc71;font-size:11px;font-weight:bold;margin-bottom:3px;">③ 자가 관리</div>'
        f'<div style="background:#f0f4ff;border-left:3px solid #2ecc71;border-radius:0 4px 4px 0;'
        f'padding:7px 10px;font-size:12px;color:#1e293b;">{steps["step3"]}</div></div>'
        # STEP 4 경고
        f'<div style="margin-bottom:8px;">'
        f'<div style="color:#dc2626;font-size:11px;font-weight:bold;margin-bottom:3px;">⚠️ 이 증상 시 즉시 내원</div>'
        f'<ul style="background:#fff0f0;border-left:3px solid #ff7070;border-radius:0 4px 4px 0;'
        f'padding:7px 10px 7px 24px;font-size:12px;margin:0;">{warn_items}</ul></div>'
        # 일정
        f'<div style="background:#eef2ff;border-radius:6px;padding:8px 10px;font-size:12px;">'
        f'<span style="color:#1e293b;">📅 {now}</span>&nbsp;&nbsp;'
        f'<span style="color:{sched_col};font-weight:bold;">{schedule}</span>'
        f'<div style="color:#1e293b;font-size:11px;margin-top:2px;">권장 조치: {info["action"]}</div></div>'
        f'<div style="margin-top:8px;font-size:10px;color:#1e293b;text-align:center;">'
        f'⚠️ AI 보조진단 — 수의사 최종 진단을 반드시 받으세요</div>'
        f'</div>'
    )

# ── 경증/중증 차이 + 금기 음식 데이터 (5-class) ──
_SEVERITY_DIFF = {
    0: {  # 태선화
        'mild':  ['피부 약간 두꺼워짐, 초기 색 변화', '국소 부위 핥기·긁기 반복', '아직 가역적 변화 단계'],
        'severe':['심한 두께 증가, 광범위 흑색 색소침착', '만성화로 정상 피부 회복 어려움', '아토피·자가면역 진행 가능성'],
        'foods': [('알레르기 단백질', '닭·소·밀 등 — 개별 과민 원인 파악 필요'), ('고당분 간식', '염증성 사이토카인 증가'), ('글루텐 함유 사료', '장-피부 축 염증 유발'), ('인공 향료·색소', '면역 반응 자극')],
    },
    1: {  # 농포/여드름
        'mild':  ['소수의 작은 농포, 국소적·얕은 위치', '발적 경미, 통증 없거나 약함', '항생제 없이 호전될 수 있음'],
        'severe':['다발성 농포, 전신 확산·깊은 감염', '발열·무기력·식욕 저하 동반', '림프절 종창, 패혈증 위험'],
        'foods': [('고당분 간식·과일', '세균 번식 환경 조성'), ('날생선·날고기', '살모넬라 등 2차 감염 위험'), ('발효식품·치즈', '면역 교란 가능'), ('고지방 트릿', '염증 반응 촉진')],
    },
    2: {  # 미란/궤양
        'mild':  ['표피만 손상된 얕은 미란, 면적 작음', '출혈 없거나 극히 경미', '청결 유지 시 자연 치유 가능'],
        'severe':['진피·피하까지 깊은 궤양, 면적 넓음', '지속 출혈·괴사 조직, 심한 악취', '패혈증·전신 감염 위험, 즉시 처치 필요'],
        'foods': [('딱딱한 건식 사료', '상처 부위 자극 및 세균 유입 경로'), ('날음식', '상처 통한 감염 위험 극대화'), ('고염분·절임 식품', '상처 삼투압 변화로 치유 방해'), ('뼈 간식', '상처 부위 직접 자극')],
    },
    3: {  # 결절/종괴
        'mild':  ['작고 부드러운 덩어리, 이동성 있음', '성장 매우 느림, 통증 없음', '양성 낭종·지방종 가능성 높음'],
        'severe':['빠른 크기 증가, 주변 조직과 유착', '궤양화·출혈·악취, 표면 변색', '악성 종양(암) 가능성 — 즉시 조직검사 필요'],
        'foods': [('가공육·햄·소시지', '아질산염 등 발암 물질 함유'), ('고지방 고칼로리식', '비만 → 종양 위험 증가'), ('인공 방부제 BHA/BHT', '발암 가능성 물질'), ('고당분 간식', '암세포 성장 연료 제공 가능')],
    },
    4: {  # 정상
        'mild':  ['현재 피부 이상 없음 — 예방적 관리 중요', '정기 검진(3~6개월)으로 조기 발견 가능', '생활 환경·식이 관리로 건강 유지'],
        'severe':['증상 없더라도 갑작스러운 변화 주의', '스트레스·환경 변화 시 면역력 저하 가능', '이상 소견 발생 시 즉시 재촬영 권장'],
        'foods': [('포도·건포도', '신부전 유발 — 소량도 치명적'), ('초콜릿·카카오', '테오브로민 독성 — 심장 이상·경련'), ('양파·마늘·대파', '적혈구 파괴 — 용혈성 빈혈'), ('자일리톨', '저혈당 쇼크 — 무설탕 제품 주의')],
    },
}


def h_disease_detail(dis_id):
    """경증/중증 증상 차이 + 금기 음식 섹션 HTML"""
    data  = _SEVERITY_DIFF[dis_id]
    info  = DISEASE_INFO[dis_id]
    mild  = data['mild']
    sev   = data['severe']
    foods = data['foods']

    mild_items = ''.join(
        f'<li style="margin:4px 0;color:#166534;">{t}</li>' for t in mild)
    sev_items  = ''.join(
        f'<li style="margin:4px 0;color:#991b1b;">{t}</li>' for t in sev)

    food_cards = ''
    for name, reason in foods:
        food_cards += (
            f'<div style="background:#fff0f0;border-left:3px solid #f44336;border-radius:0 6px 6px 0;'
            f'padding:6px 10px;margin-bottom:6px;">'
            f'<div style="color:#dc2626;font-weight:bold;font-size:12px;">🚫 {name}</div>'
            f'<div style="color:#1e293b;font-size:11px;margin-top:2px;">{reason}</div>'
            f'</div>'
        )

    return (
        f'<div style="background:#f8fafc;border-radius:8px;padding:14px;margin-top:8px;">'
        # 제목
        f'<div style="color:#1a2a3a;font-weight:bold;font-size:14px;margin-bottom:12px;">'
        f'📖 {info["icon"]} {info["name"]} — 증상 단계별 차이 & 주의 음식</div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;">'

        # 왼쪽: 경증 vs 중증
        f'<div style="flex:2;min-width:260px;">'
        f'<div style="display:flex;gap:8px;">'
        # 경증
        f'<div style="flex:1;background:#f0fdf4;border-radius:8px;padding:10px 12px;">'
        f'<div style="color:#4CAF50;font-weight:bold;font-size:12px;margin-bottom:6px;">🟢 경증 (가정 관찰 가능)</div>'
        f'<ul style="margin:0;padding-left:16px;font-size:12px;line-height:1.8;">{mild_items}</ul>'
        f'</div>'
        # 중증
        f'<div style="flex:1;background:#fff0f0;border-radius:8px;padding:10px 12px;">'
        f'<div style="color:#F44336;font-weight:bold;font-size:12px;margin-bottom:6px;">🔴 중증 (즉시 내원 필요)</div>'
        f'<ul style="margin:0;padding-left:16px;font-size:12px;line-height:1.8;">{sev_items}</ul>'
        f'</div>'
        f'</div></div>'

        # 오른쪽: 금기 음식
        f'<div style="flex:1;min-width:200px;">'
        f'<div style="color:#dc2626;font-weight:bold;font-size:12px;margin-bottom:8px;">⚠️ 이 증상일 때 자제할 음식</div>'
        f'{food_cards}'
        f'</div>'

        f'</div>'  # flex row
        f'<div style="color:#1e293b;font-size:10px;margin-top:8px;text-align:right;">'
        f'수의학 일반 지침 기반 | 개별 반려동물 상태에 따라 다를 수 있습니다</div>'
        f'</div>'
    )


def h_all_disease_bars(probs=None):
    """6개 질환별 AI 예측 확률 막대 표시"""
    if probs is None:
        return ('<div style="background:#f8fafc;border-radius:8px;padding:14px;'
                'text-align:center;color:#1e293b;font-size:12px;">분석 후 표시됩니다</div>')
    bars = ''
    for i in range(NUM_CLASSES):
        pct  = float(probs[i]) * 100
        col  = DISEASE_COLORS_LIST[i]
        icon = DISEASE_INFO[i]['icon']
        name = DISEASE_NAMES_KR[i]
        urg  = DISEASE_INFO[i]['urgency']
        badge_col = '#4CAF50' if urg == 0 else ('#FF9800' if urg == 1 else '#F44336')
        badge_txt = '정상' if urg == 0 else ('주의' if urg == 1 else '긴급')
        bars += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
            f'<div style="width:18px;text-align:center;font-size:14px;">{icon}</div>'
            f'<div style="width:72px;font-size:11px;color:#1e293b;white-space:nowrap;">{name}</div>'
            f'<div style="flex:1;background:#f8fafc;border-radius:4px;height:16px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;background:{col};height:100%;border-radius:4px;min-width:2px;"></div></div>'
            f'<div style="width:36px;font-size:11px;color:{col};font-weight:bold;text-align:right;">{pct:.1f}%</div>'
            f'<span style="background:{badge_col};color:white;font-size:9px;padding:1px 6px;'
            f'border-radius:10px;white-space:nowrap;">{badge_txt}</span>'
            f'</div>'
        )
    return (f'<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;">'
            f'<div style="color:#1e293b;font-size:11px;font-weight:bold;margin-bottom:8px;">🧬 질환별 AI 예측 확률</div>'
            f'{bars}</div>')


# ── SigLIP 2 분류 헤드 (6-class) ──
class DiseaseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(1152),
            nn.Linear(1152, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def load_models():
    M = {}

    # YOLO26m
    if YOLO_W.exists():
        from ultralytics import YOLO
        M['yolo'] = YOLO(str(YOLO_W)); print("YOLO26m loaded")
    else:
        M['yolo'] = None; print(f"YOLO not found: {YOLO_W}")

    # SigLIP 2 SO400M 6-class 파인튜닝
    try:
        print(f"Loading SigLIP 2 ({SIGLIP2_ID})...")
        M['processor'] = AutoProcessor.from_pretrained(SIGLIP2_ID)
        full_model     = AutoModel.from_pretrained(SIGLIP2_ID)
        encoder        = full_model.vision_model.to(DEVICE)

        # 6-class 체크포인트 적용 (fp16 저장 → float32 변환)
        ckpt    = torch.load(str(SIGLIP2_FT_W), map_location=DEVICE)
        enc_sd  = encoder.state_dict()
        enc_upd = {k: v.float() for k, v in ckpt['encoder_last_blocks'].items()}
        enc_sd.update(enc_upd)
        encoder.load_state_dict(enc_sd)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        head = DiseaseHead().to(DEVICE)
        head.load_state_dict(ckpt['head'])
        head.eval()

        M['encoder'] = encoder
        M['head']    = head
        print(f"SigLIP 2 FT loaded (val_acc={ckpt.get('val_acc', 0):.4f}) ★")
    except Exception as e:
        M['encoder'] = M['head'] = M['processor'] = None
        print(f"SigLIP 2 load failed: {e}")

    # SAM2.1 (세그멘테이션, 학습 불필요)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if SAM2_W.exists():
            M['sam2'] = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", str(SAM2_W), device=DEVICE))
            print("SAM2.1 loaded")
        else:
            M['sam2'] = None
    except Exception as e:
        M['sam2'] = None; print(f"SAM2 not available: {e}")
    return M


# ── 전처리 ──
def _apply_clahe(img_rgb):
    """훈련과 동일한 CLAHE 전처리 (RGB 입력/출력)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _get_crop(img_rgb, bbox):
    h, w = img_rgb.shape[:2]
    if bbox is not None:
        x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        px=int((x2-x1)*.15); py=int((y2-y1)*.15)
        x1=max(0,x1-px); y1=max(0,y1-py); x2=min(w,x2+px); y2=min(h,y2+py)
        if x2>x1 and y2>y1: return img_rgb[y1:y2, x1:x2]
    # YOLO 미탐지 시 중앙 70% crop (손·배경 제거)
    ph, pw = int(h * 0.15), int(w * 0.15)
    return img_rgb[ph:h-ph, pw:w-pw]

def classify(img_rgb, bbox):
    crop = _get_crop(img_rgb, bbox)
    crop = _apply_clahe(crop)

    if MODELS.get('encoder') is None:
        probs = np.zeros(NUM_CLASSES, dtype=np.float32); probs[NORMAL_IDX] = 1.0
        return NORMAL_IDX, 0, '모델 미로드', probs

    # TTA: 원본 + 좌우반전
    augs = [crop, crop[:, ::-1].copy()]
    processor = MODELS['processor']
    with torch.no_grad():
        inputs = processor(images=augs, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feats  = MODELS['encoder'](**inputs).pooler_output   # (2, 1152)
        logits = MODELS['head'](feats)                       # (2, 7)
        probs  = torch.softmax(logits, 1).mean(0).cpu().numpy().astype(np.float32)

    pred_id     = int(probs.argmax())
    safety_warn = ''
    if pred_id == NORMAL_IDX and float(probs[NORMAL_IDX]) < NORMAL_CONF_THRESH:
        pred_id     = int(probs[:NORMAL_IDX].argmax())
        safety_warn = f'⚠️ 정상 경계({probs[NORMAL_IDX]*100:.0f}%)→의심 질환 표시'

    tta_info = f"SigLIP2 TTA×2 | {probs[pred_id]*100:.0f}% 확신도"
    if safety_warn:
        tta_info += f" | {safety_warn}"
    return pred_id, 0, tta_info, probs


# ── 메인 파이프라인 ──
def run_analysis(image, pet_name):
    empty_dis = h_disease_card(NORMAL_IDX)
    if image is None:
        return (None, make_gauge(0), h_alert(0),
                h_patient('미분석','미분석'), h_guidance(0, NORMAL_IDX), h_all_disease_bars(None),
                empty_dis, h_ai_opinion(), h_disease_detail(NORMAL_IDX))

    _state['pet_name'] = pet_name.strip() if pet_name and pet_name.strip() else '반려동물'
    img_rgb = np.array(image)
    out = img_rgb.copy()
    bbox = None

    # Stage 1: YOLO26m
    if MODELS.get('yolo'):
        res = MODELS['yolo'](img_rgb, verbose=False)
        if res and len(res[0].boxes) > 0:
            bidx = res[0].boxes.conf.argmax().item()
            box  = res[0].boxes.xyxy[bidx].cpu().numpy().astype(int)
            bbox = box
            cv2.rectangle(out,(box[0],box[1]),(box[2],box[3]),(0,230,100),3)
            cv2.putText(out,'lesion',(box[0],max(0,box[1]-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(0,230,100),2)

    # Stage 2: SAM2.1
    if MODELS.get('sam2') and bbox is not None:
        try:
            pred = MODELS['sam2']
            pred.set_image(img_rgb)
            ms,_,_ = pred.predict(point_coords=None,point_labels=None,
                                   box=np.array(bbox[:4])[None,:],multimask_output=False)
            if ms is not None and len(ms)>0:
                mask = ms[0].astype(bool)
                ov = out.copy()
                ov[mask] = (ov[mask]*.4+np.array([0,200,100])*.6).astype(np.uint8)
                out = ov
        except Exception as e:
            print(f"SAM2 err: {e}")

    # Stage 3: SigLIP 2 SO400M (74.24%, TTA×2)
    dis_id, reg_id, tta_info, probs = classify(img_rgb, bbox)
    _state['dis_id'] = dis_id; _state['reg_id'] = reg_id
    _state['probs']  = probs;  _state['image']  = img_rgb

    final = disease_score(dis_id, probs)
    dis_name = DISEASE_INFO[dis_id]['name']
    reg_name = REGION_KR[reg_id]
    conf_pct = f"{probs[dis_id]*100:.0f}% 확신도"

    # AI 소견 생성
    main_t, action_t, ddx_h, conf_col, conf_p = generate_ai_opinion(img_rgb, dis_id, probs)

    return (out,
            make_gauge(final),
            h_alert(final),
            h_patient(dis_name, reg_name, tta_info),
            h_guidance(final, dis_id),
            h_all_disease_bars(probs),
            h_disease_card(dis_id, conf_pct, probs),
            h_ai_opinion(main_t, action_t, ddx_h, conf_col, conf_p),
            h_disease_detail(dis_id))


def update_score(ulcer, erythema, scale, protect):
    dis_id = _state.get('dis_id', NORMAL_IDX)
    probs  = _state.get('probs')
    final, ulcer_c, erythema_c, scale_c = calc_score(ulcer, erythema, scale, protect)
    return (make_gauge(final), h_alert(final),
            h_stats(final, erythema_c, scale_c, ulcer_c, probs),
            h_guidance(final, dis_id),
            h_all_disease_bars(probs),
            h_all_disease_bars(probs))

def reset_sliders():
    sl = DISEASE_SLIDERS.get(_state.get('dis_id', NORMAL_IDX), DISEASE_SLIDERS[0])
    return [gr.update(value=v) for v in sl]


CSS = """
body, .gradio-container { background:#eef2ff !important; color:#1e293b !important; }
.block, .form, .panel  { background:#ffffff !important; border:1px solid #e2e8f0 !important; border-radius:12px !important; box-shadow:0 1px 4px rgba(0,0,0,0.06) !important; }
label, .label-wrap span { color:#374151 !important; font-size:12px !important; font-weight:600 !important; }
button { background:#2563eb !important; color:white !important; border:none !important; border-radius:8px !important; cursor:pointer; font-weight:600 !important; }
button.primary { background:#1d4ed8 !important; }
button:hover   { opacity:0.88 !important; }
input[type=range] { accent-color:#2563eb !important; }
footer, .footer { display:none !important; }
#input_img, #result_img,
#input_img > div, #result_img > div,
#input_img .image-container, #result_img .image-container,
#input_img .upload-container, #result_img .upload-container,
#input_img [data-testid="image"], #result_img [data-testid="image"],
#input_img button, #result_img button { background:#ffffff !important; }
#input_img svg *, #result_img svg * { stroke:#94a3b8 !important; fill:none !important; }
#input_img *, #result_img * { color:#000000 !important; }
"""

print("Loading models...")
MODELS = load_models()

with gr.Blocks() as demo:

    # ── 헤더 ──
    gr.HTML(
        '<div style="padding:10px 0 6px;">'
        '<div style="font-size:22px;font-weight:bold;color:#1a2a3a;">🐾 반려동물 피부 질환 AI 진단</div>'
        '<div style="color:#64748b;font-size:12px;margin-top:2px;">'
        'YOLO26m → SAM2.1 → SigLIP 2 (2025) | 사진 업로드 후 질환을 자동 분류합니다</div></div>')

    alert_box = gr.HTML(h_alert(0))

    # ── 진단 결과 카드 (전체 너비) ──
    disease_card = gr.HTML(h_disease_card(NORMAL_IDX))

    # ── 메인 3열 ──
    with gr.Row():
        with gr.Column(scale=3, min_width=260):
            gr.HTML('<div style="color:#1a2a3a;font-weight:bold;font-size:13px;margin-bottom:4px;">📷 업로드 이미지</div>')
            inp     = gr.Image(label="", type="pil", height=260, elem_id="input_img")
            res_img = gr.Image(label="탐지 결과", height=260, visible=False, elem_id="result_img")
            pet_in  = gr.Textbox(label="반려동물 이름", placeholder="예: 코코", value="")
            run_btn = gr.Button("🔍 분석 시작", variant="primary")
            pat_box = gr.HTML(h_patient('미분석','미분석'))

        with gr.Column(scale=4, min_width=320):
            gauge_p = gr.Plot(show_label=False)
            dis_box = gr.HTML(h_all_disease_bars(None))

        with gr.Column(scale=3, min_width=240):
            guide_box   = gr.HTML(h_guidance(0, NORMAL_IDX))
            opinion_box = gr.HTML(h_ai_opinion())

    # ── 경증/중증 + 금기음식 ──
    detail_box = gr.HTML(h_disease_detail(NORMAL_IDX))

    # ── 병원 찾기 ──
    with gr.Row():
        loc_btn  = gr.Button("📍 내 위치 기반 동물병원 찾기", variant="secondary")
    lat_in       = gr.Textbox(visible=False, elem_id="lat_in", value="")
    lon_in       = gr.Textbox(visible=False, elem_id="lon_in", value="")
    hospital_box = gr.HTML('')

    gr.HTML('<div style="text-align:center;color:#374151;font-size:11px;margin-top:10px;">'
            '⚠️ 본 시스템은 AI 보조진단 목적으로만 사용되며, 수의사의 최종 진단을 대체하지 않습니다</div>')

    # ── 이벤트 ──
    def _run(image, pet_name):
        results = run_analysis(image, pet_name)
        img_out = results[0]
        rest    = results[1:]
        return (gr.update(visible=False),
                gr.update(value=img_out, visible=True),
                *rest)

    def search_hospitals(lat_str, lon_str):
        try:
            lat, lon = float(lat_str), float(lon_str)
            hospitals = find_nearby(lat, lon, n=5)
            dis_id    = _state.get('dis_id', 6)
            html      = h_hospital_section(dis_id, hospitals)
            if not html:
                return ('<div style="background:#f8fafc;border-radius:8px;padding:14px;'
                        'color:#f44;font-size:13px;">근처 병원 데이터를 찾을 수 없습니다.</div>')
            return html
        except Exception as e:
            return (f'<div style="background:#f8fafc;border-radius:8px;padding:14px;'
                    f'color:#f44;font-size:13px;">위치 정보를 가져올 수 없습니다: {e}</div>')

    # JS: 비동기 geolocation → [lat, lon] 반환 → Python search_hospitals 로 직접 전달
    _GEO_JS = """async () => {
        return new Promise((resolve) => {
            if (!navigator.geolocation) {
                resolve(["", ""]);
                return;
            }
            navigator.geolocation.getCurrentPosition(
                (pos) => resolve([String(pos.coords.latitude), String(pos.coords.longitude)]),
                ()    => resolve(["", ""])
            );
        });
    }"""

    run_btn.click(
        _run,
        inputs=[inp, pet_in],
        outputs=[inp, res_img, gauge_p, alert_box,
                 pat_box, guide_box, dis_box, disease_card, opinion_box, detail_box])

    loc_btn.click(
        fn=search_hospitals,
        inputs=[lat_in, lon_in],
        outputs=[hospital_box],
        js=_GEO_JS)


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True,
                theme=gr.themes.Soft(), css=CSS)
