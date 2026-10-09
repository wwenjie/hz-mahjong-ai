// 牌码 → 显示信息。内部牌码：1w-9w(万) / 1b-9b(筒) / 1t-9t(条) / 东南西北中发白。
const SUIT_LABEL = { w: '万', b: '筒', t: '条', 万: '万', 筒: '筒', 条: '条' }

export function tileInfo(code) {
  if (!code) return { num: '', suit: '', honor: true, label: '' }
  if (code.length === 2 && /[1-9]/.test(code[0])) {
    const n = code[0]
    const s = SUIT_LABEL[code[1]] || code[1]
    return { num: n, suit: s, honor: false, label: n + s }
  }
  return { num: '', suit: code, honor: true, label: code }
}

export function isGod(code) {
  return code === '白'
}

// 把牌码数组按稳定顺序排序（万→筒→条→字）
const SUIT_ORDER = { w: 0, b: 1, t: 2 }
export function sortCodes(codes) {
  return [...codes].sort((a, b) => codeKey(a) - codeKey(b))
}
// 单张牌的排序键（导出，供拖拽插入新牌时定位）
export function tileOrderKey(code) {
  return codeKey(code)
}
function codeKey(code) {
  if (code.length === 2 && /[1-9]/.test(code[0])) {
    const s = SUIT_ORDER[code[1]] ?? 9
    return s * 100 + Number(code[0])
  }
  const honors = ['东', '南', '西', '北', '中', '发', '白']
  return 300 + honors.indexOf(code)
}

export function phaseLabel(phase) {
  return (
    {
      draw: '轮到你出牌',
      response_peng: '可碰/杠',
      response_chi: '可吃',
      deal: '发牌',
      settled: '本局结算',
      finished: '结束',
    }[phase] || phase
  )
}
