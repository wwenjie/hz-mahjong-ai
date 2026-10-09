<script setup>
import { computed, defineComponent, h, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import Tile from './Tile.vue'
import { phaseLabel, sortCodes, tileInfo, tileOrderKey } from './tiles.js'

// 决策档：由后端 /api/state 下发（v7=当前线上采集档）；标题/座位名/建议栏据此显示。
const decider = ref('v7')

const state = reactive({
  status: 'connecting',
  finished: false,
  humanSeat: 0,
  totalRounds: 8,
  view: null,
  required: null,
  rounds: [],
  cumulative: [0, 0, 0, 0],
  log: [],
})

const errorMsg = ref('')
const busy = ref(false)
const reveal = ref(false)
// 建议：开关 + 最近一次建议（来自后端 suggestion 事件 / /api/state）
const suggestEnabled = ref(true)
const suggestion = ref(null)
// 手动续局：上局已结束、等点「开下一局」
const awaitingNext = ref(false)
const nextRoundNo = ref(null)
// 上一局结果（用于结果面板）
const roundResult = ref(null)

// 报障：保存当前局面 + 引擎建议 + 备注，供离线复现（不需要一直保持场景）
const reportOpen = ref(false)
const reportText = ref('')
const reportBusy = ref(false)
const reportDone = ref('')
const reportError = ref('')

// --------------------------------------------------------------------------- //
// 手牌顺序：每局自动按花色排序一次；局内允许拖拽微调；重开/新局时重置为自动排序
// --------------------------------------------------------------------------- //
const handOrder = ref([]) // 牌码数组，代表显示顺序

function resetHandOrder(codes) {
  handOrder.value = sortCodes(codes)
}

// 把新到的牌并入现有顺序（未排序过就自动排序）
function syncHand(codes) {
  const pool = [...codes]
  const kept = []
  for (const c of handOrder.value) {
    const i = pool.indexOf(c)
    if (i >= 0) {
      kept.push(c)
      pool.splice(i, 1)
    }
  }
  // 剩余的是新摸到的牌：按牌序插入
  for (const c of sortCodes(pool)) {
    const key = tileOrderKey(c)
    const at = kept.findIndex((x) => tileOrderKey(x) > key)
    if (at < 0) kept.push(c)
    else kept.splice(at, 0, c)
  }
  handOrder.value = kept
}

// 拖拽期间本地临时顺序（仅用于渲染预览）
const dragOrder = ref(null)
const drag = reactive({ active: false, code: null, from: -1, over: -1 })

// --------------------------------------------------------------------------- //
// 自适应：测量牌的**真实尺寸**，按视口缩放；容器高度随之设定，居中不偏移
// --------------------------------------------------------------------------- //
const minZoom = ref(0.6)
const maxZoom = ref(3.0)

try {
  const q = new URLSearchParams(window.location.search)
  if (q.get('zmin')) minZoom.value = Number(q.get('zmin'))
  if (q.get('zmax')) maxZoom.value = Number(q.get('zmax'))
} catch {
  /* ignore */
}

const stageRef = ref(null)
const zoom = ref(1)
const autoZoom = ref(1)
const zoomBias = ref(1)
const natW = ref(980)
const natH = ref(760)
// 右侧「对局记录」栏 + 页面留白占用的总水平空间：缩放时须预留，否则会与牌桌重叠
const LOG_W = 320

function applyZoom() {
  const z = autoZoom.value * zoomBias.value
  zoom.value = Math.max(minZoom.value, Math.min(maxZoom.value, z))
}
function recomputeZoom() {
  const el = stageRef.value
  if (!el) return
  // offsetWidth / scrollHeight 不受 transform 影响，得到的即“真实尺寸”
  const w = el.offsetWidth || 980
  const h = el.scrollHeight || 760
  natW.value = w
  natH.value = h
  const availW = window.innerWidth - LOG_W
  const availH = window.innerHeight - 30
  let z = Math.min(availW / w, availH / h)
  if (!isFinite(z) || z <= 0) z = 1
  autoZoom.value = z
  applyZoom()
}
function zoomIn() { zoomBias.value = Math.min(4, +(zoomBias.value + 0.1).toFixed(2)); applyZoom() }
function zoomOut() { zoomBias.value = Math.max(0.4, +(zoomBias.value - 0.1).toFixed(2)); applyZoom() }

// 缩放系数与容器尺寸
const stageStyle = computed(() => ({
  width: natW.value + 'px',
  transform: `scale(${zoom.value})`,
  transformOrigin: 'top left',
}))
// 外层盒：按「缩放后」的真实尺寸占位，避免缩放后的牌桌与右侧记录栏重叠
const boxStyle = computed(() => ({
  width: natW.value * zoom.value + 'px',
  height: natH.value * zoom.value + 'px',
}))

// --------------------------------------------------------------------------- //
// 座位相对位置：以人类为下方(base)，逆时针(0→1→2→3) → 右手边是下家
// --------------------------------------------------------------------------- //
const relSeat = (seat) => (seat - state.humanSeat + 4) % 4
const seatsByPos = computed(() => {
  const out = { bottom: null, right: null, top: null, left: null }
  if (!state.view) return out
  const pos = ['bottom', 'right', 'top', 'left']
  for (let s = 0; s < 4; s++) out[pos[relSeat(s)]] = s
  return out
})
const relNames = computed(() => {
  const m = {}
  const names = ['你', decider.value + '·下家', decider.value + '·对家', decider.value + '·上家']
  for (let s = 0; s < 4; s++) m[s] = names[relSeat(s)]
  return m
})

function seatData(seat) {
  const v = state.view
  if (!v || seat === null || seat === undefined) return {}
  const revealed = Array.isArray(v.others_hands) ? v.others_hands[seat] : null
  return {
    name: relNames.value[seat],
    isDealer: v.dealer_seat === seat,
    isTurn: v.turn === seat,
    handCount: v.hand_counts?.[seat] ?? 13,
    score: state.cumulative?.[seat] ?? 0,
    hand: seat === state.humanSeat ? null : revealed,
    melds: v.melds?.[seat] || [],
  }
}
const poolOf = (seat) => (seat === null || seat === undefined ? [] : state.view?.discards?.[seat] || [])

const myHand = computed(() => {
  const codes = state.view ? state.view.my_hand || [] : []
  const ordered = handOrder.value
  return dragOrder.value || (ordered.length ? keptOnly(ordered, codes) : sortCodes(codes))
})

// 显示用的定位键：优先用稳定下标（运行时重建，不入后端数据），
// 避免重复牌（两张 8b）的 :key 冲突导致 Vue 复用错元素。
const myHandKeyed = computed(() =>
  myHand.value.map((code, i) => ({ code, key: dragOrder.value ? `d${i}` : `c${i}` })),
)

// 只保留当前手牌里仍存在的牌，避免旧顺序残留
function keptOnly(ordered, codes) {
  const pool = [...codes]
  const out = []
  for (const c of ordered) {
    const i = pool.indexOf(c)
    if (i >= 0) {
      out.push(c)
      pool.splice(i, 1)
    }
  }
  for (const c of sortCodes(pool)) out.push(c)
  return out
}
const myMelds = computed(() => state.view?.my_melds || [])
const mySeat = computed(() => state.humanSeat)

const selected = ref(null)
const discardActions = computed(() =>
  state.required ? state.required.actions.filter((a) => a.kind === 'discard') : [],
)
const otherActions = computed(() =>
  state.required ? state.required.actions.filter((a) => a.kind !== 'discard') : [],
)
// 当前是否轮到我出牌（有 discard 选项）
const canDiscard = computed(() => discardActions.value.length > 0)
const canDiscardCode = (code) => discardActions.value.some((a) => a.tile === code)

// 响应阶段（别人打牌、我能碰/杠/吃）
const isResponse = computed(() => (state.required?.phase || '').startsWith('response'))
const offeredTile = computed(() => state.required?.view?.offered_tile || null)
const offeredBy = computed(() => {
  const v = state.required?.view
  if (!v || !v.offered_tile) return ''
  const seat = v.turn
  return seat === state.humanSeat ? '' : relNames.value[seat] || ''
})
const responseVerb = computed(() => {
  const kinds = otherActions.value.map((a) => a.kind)
  const map = { peng: '碰', gang: '杠', chi: '吃', hu: '胡', pass: '过' }
  const acts = kinds.filter((k) => k !== 'pass').map((k) => map[k] || k)
  return acts.length ? acts.join(' / ') : ''
})
const tileLabel = (code) => tileInfo(code).label

// 吃牌窗口：吃是「用手里两张凑顺子」，可能有多种吃法
const isChi = computed(() => (state.required?.phase || '') === 'response_chi')
const chiCombos = computed(() =>
  otherActions.value.filter((a) => a.kind === 'chi').map((a) => a.tiles || []),
)

// 动作按钮文案
const ACTION_LABEL = { pass: '过', peng: '碰', gang: '杠', hu: '胡', chi: '吃' }
function actionLabel(a) {
  return ACTION_LABEL[a.kind] || a.kind
}
function actionHint(a) {
  if (a.kind === 'chi') return (a.tiles || []).map(tileLabel).join(' ')
  if (a.kind === 'peng' || a.kind === 'gang') return a.tile ? tileLabel(a.tile) : ''
  return ''
}

// 自摸成胡（本引擎只支持自摸胡，点炮胡/抢杠被平台禁止）
const canHu = computed(() => otherActions.value.some((a) => a.kind === 'hu'))

// 只采纳「与当前决策点同序号」的建议——避免慢计算的上一条建议被误当当前建议。
const activeSuggestion = computed(() => {
  const s = suggestion.value
  if (!suggestEnabled.value || !s) return null
  const seq = state.required?.seq
  if (seq == null) return null
  return s.seq === seq ? s : null
})
// 建议卡片文案：把「引擎在当前决策点的推荐」渲染成人话。
// 注意：正前方的标签（下方模板 ``s-tag``）已经写了「{版本} 建议」，
// 因此这里的正文**不再重复**「建议」二字，只描述动作。
const suggestText = computed(() => {
  if (!suggestEnabled.value) return ''
  const s = activeSuggestion.value
  if (!s) return state.required ? '计算中…' : ''
  if (!s.action) return '本轮无可行动作'
  const a = s.action
  const kind = ACTION_LABEL[a.kind] || a.kind
  if (a.kind === 'discard') return '打出 ' + tileLabel(a.tile)
  if (a.kind === 'chi') return '吃（用 ' + (a.tiles || []).map(tileLabel).join('+') + '）'
  if (a.kind === 'peng') return '碰 ' + tileLabel(a.tile)
  if (a.kind === 'gang') return '杠 ' + tileLabel(a.tile)
  if (a.kind === 'hu') return '胡'
  return kind
})
// 建议是否指向「出某张牌」——用于在手牌上高亮
const suggestDiscardCode = computed(() => {
  const s = activeSuggestion.value
  if (!s || !s.action) return null
  return s.action.kind === 'discard' ? s.action.tile : null
})
// 建议是否与某个可点动作一致（碰/吃/胡/过）——用于给按钮加标记
function isSuggested(a) {
  const s = activeSuggestion.value
  if (!s || !s.action) return false
  const s2 = s.action
  if (a.kind !== s2.kind) return false
  if (a.kind === 'chi') return JSON.stringify(a.tiles || []) === JSON.stringify(s2.tiles || [])
  return true
}

// 本回合摸到的牌（仅出牌阶段、且确实摸牌时才有；碰/吃后补轮为 null）
const drawnTile = computed(() => {
  if (!state.required || state.required.phase !== 'draw') return null
  return state.required.view?.drawn_tile || null
})
const drawnIndex = computed(() => {
  const t = drawnTile.value
  if (!t) return -1
  // 摸到的牌插在最右：取最右侧的匹配下标（重复牌时不能取第一张）
  return myHand.value.lastIndexOf(t)
})

function confirmDiscard() {
  if (selected.value && canDiscardCode(selected.value)) {
    submit({ kind: 'discard', tile: selected.value })
  }
}

// 拖拽排序 ------------------------------------------------------------------- //
// 关键：全程按**下标**操作，绝不按牌码（重复牌会拖错）。
function onDragStart(code, index, ev) {
  drag.active = true
  drag.code = code
  drag.from = index
  drag.over = index
  dragOrder.value = [...myHand.value]
  try {
    ev.dataTransfer.effectAllowed = 'move'
    ev.dataTransfer.setData('text/plain', code)
  } catch {
    /* ignore */
  }
}

function onDragOver(index, ev) {
  if (!drag.active) return
  ev.preventDefault()
  drag.over = index
  // 实时预览：用「被拖牌在临时数组里的当前下标」定位，避免重复牌找错。
  const arr = [...(dragOrder.value || myHand.value)]
  const from = arr.indexOf(drag.code)
  if (from < 0) return
  const [c] = arr.splice(from, 1)
  const to = Math.min(index, arr.length)
  arr.splice(to, 0, c)
  dragOrder.value = arr
}

function onDragEnd() {
  if (drag.active && dragOrder.value) {
    handOrder.value = [...dragOrder.value]
  }
  drag.active = false
  drag.code = null
  drag.from = -1
  drag.over = -1
  dragOrder.value = null
}

// --------------------------------------------------------------------------- //
// 小组件：弃牌组 / 副露组 / 翻开的手牌 / 座位牌
// --------------------------------------------------------------------------- //
const Discards = defineComponent({
  props: { tiles: Array, vertical: Boolean },
  setup(props) {
    return () =>
      h(
        'div',
        { class: ['discards', props.vertical ? 'col' : 'row'] },
        (props.tiles || []).map((c, i) => h(Tile, { code: c, mini: true, key: i })),
      )
  },
})

const Melds = defineComponent({
  props: { melds: Array, vertical: Boolean },
  setup(props) {
    return () =>
      h(
        'div',
        { class: ['melds', props.vertical ? 'col' : 'row'] },
        (props.melds || []).map((m, mi) =>
          h(
            'div',
            { class: 'meld', key: mi },
            m.tiles.map((c, ti) => h(Tile, { code: c, small: true, key: ti })),
          ),
        ),
      )
  },
})

const RevHand = defineComponent({
  props: { tiles: Array },
  setup(props) {
    return () =>
      h(
        'div',
        { class: 'revhand' },
        (props.tiles || []).map((c, i) => h(Tile, { code: c, mini: true, ghost: true, key: i })),
      )
  },
})

const SeatCard = defineComponent({
  props: {
    name: String,
    isDealer: Boolean,
    isTurn: Boolean,
    handCount: Number,
    score: Number,
    hand: Array,
    melds: Array,
    vertical: Boolean,
    hideMelds: Boolean,
  },
  setup(props) {
    return () =>
      h('div', { class: ['seatcard', { turn: props.isTurn, v: props.vertical }] }, [
        h('div', { class: 'nameplate' }, [
          h('span', { class: 'pname' }, props.name),
          props.isDealer ? h('span', { class: 'dealer' }, '庄') : null,
          h('span', { class: ['belly', { neg: props.score < 0 }] }, String(props.score)),
          h('span', { class: 'cnt' }, `${props.handCount} 张`),
        ]),
        props.hand ? h(RevHand, { tiles: props.hand }) : null,
        props.hideMelds ? null : h(Melds, { melds: props.melds, vertical: props.vertical }),
      ])
  },
})

// --------------------------------------------------------------------------- //
// 交互
// --------------------------------------------------------------------------- //
async function submit(payload) {
  if (busy.value) return
  // 记住本次提交针对的决策点：提交后的往返期间，对局可能飞速推进、
  // 新的 action_required 已经经 SSE 到达。此时不能再用成功回调清空，
  // 否则会把刚收到的决策点抹掉（按钮消失，只能刷新）。
  const pending = state.required
  busy.value = true
  errorMsg.value = ''
  try {
    const r = await fetch('/api/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const j = await r.json()
    if (!j.ok) {
      errorMsg.value = j.error || '提交失败'
      // 提交被拒通常意味着服务端已经推进/重同步（例如决策点已过期）。
      // 不要留下卡住的旧按钮：从服务端拉一次真相，恢复或清空。
      if (/没有等待中的决策点|非法动作/.test(String(j.error || ''))) {
        await hydrate(true)
      }
    } else if (state.required === pending) {
      // 期间没有新决策点到来，才清空（正常：等下一轮）
      state.required = null
      selected.value = null
    }
  } catch (e) {
    errorMsg.value = String(e)
  } finally {
    busy.value = false
  }
}

function onClickTile(code) {
  if (!state.required) return
  if (canDiscardCode(code)) selected.value = selected.value === code ? null : code
}

function doAction(a) {
  submit({ kind: a.kind, tile: a.tile, tiles: a.tiles, gang_kind: a.gang_kind })
}

function pushLog(text) {
  const t = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  state.log.unshift(`[${t}] ${text}`)
  if (state.log.length > 60) state.log.pop()
}

// --- 报障 ---------------------------------------------------------------- //
function openReport() {
  reportText.value = ''
  reportDone.value = ''
  reportError.value = ''
  reportOpen.value = true
}
function closeReport() {
  reportOpen.value = false
}
async function submitReport() {
  if (reportBusy.value) return
  reportBusy.value = true
  reportError.value = ''
  try {
    const r = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ comment: reportText.value }),
    })
    const j = await r.json()
    if (!j.ok) {
      reportError.value = j.error || '保存失败'
    } else {
      reportDone.value = j.id || '已保存'
      pushLog('📸 报障已保存 · ' + (j.id || ''))
      setTimeout(() => { if (reportOpen.value) reportOpen.value = false }, 1600)
    }
  } catch (e) {
    reportError.value = String(e)
  } finally {
    reportBusy.value = false
  }
}

function handleHand(codes) {
  if (!handOrder.value.length) resetHandOrder(codes)
  else syncHand(codes)
}

// 刷新 = 重连（不是重开）：向服务端要一次完整现状，恢复比分/记录/决策点。
// silent=true 用于「提交被拒后自愈」：不写对局记录，避免刷屏。
async function hydrate(silent = false) {
  try {
    const r = await fetch('/api/state')
    const j = await r.json()
    if (j.decider) decider.value = String(j.decider)
    state.humanSeat = j.human_seat ?? state.humanSeat
    state.totalRounds = j.total_rounds ?? state.totalRounds
    state.finished = !!j.finished
    state.cumulative = j.cumulative || [0, 0, 0, 0]
    state.rounds = j.rounds || []
    reveal.value = !!j.reveal
    if (j.decider) decider.value = String(j.decider)
    if ('suggest_enabled' in j) suggestEnabled.value = !!j.suggest_enabled
    if ('suggestion' in j) suggestion.value = j.suggestion
    awaitingNext.value = !!j.awaiting_next
    // 结果面板：若正等开下一局，用最近一局结果恢复面板；否则清空
    roundResult.value = awaitingNext.value && state.rounds.length
      ? state.rounds[state.rounds.length - 1]
      : null
    nextRoundNo.value = roundResult.value ? (roundResult.value.round_no + 1) : null
    if (j.state) {
      state.view = j.state
      if (j.state.my_hand) handleHand(j.state.my_hand)
    }
    // 当前正等人类决策：恢复按钮，别让刷新后按钮消失
    if (j.required) {
      state.required = j.required
      if (j.required.view?.my_hand) handleHand(j.required.view.my_hand)
      state.view = j.required.view
      if (!silent) pushLog('已重连 · ' + phaseLabel(j.required.phase))
    } else {
      state.required = null
      if (!silent) pushLog('已重连到进行中的对局')
    }
  } catch (e) {
    errorMsg.value = String(e)
  }
}

async function toggleReveal() {
  const next = !reveal.value
  try {
    const r = await fetch('/api/reveal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reveal: next }),
    })
    const j = await r.json()
    reveal.value = !!j.reveal
    pushLog(reveal.value ? '👁 开天眼：已翻开其余三家手牌' : '👁 关天眼：只看自己的牌')
  } catch (e) {
    errorMsg.value = String(e)
  }
}

async function toggleSuggest() {
  const next = !suggestEnabled.value
  try {
    const r = await fetch('/api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: next }),
    })
    const j = await r.json()
    suggestEnabled.value = !!j.suggest_enabled
    if (!suggestEnabled.value) suggestion.value = null
    pushLog(suggestEnabled.value ? '🧠 建议：开' : '🧠 建议：关')
  } catch (e) {
    errorMsg.value = String(e)
  }
}

async function startNextRound() {
  if (busy.value) return
  busy.value = true
  try {
    const r = await fetch('/api/next_round', { method: 'POST' })
    const j = await r.json()
    if (!j.ok) {
      // 已不在等待（例如别处已点）——以服务端为准刷新一次
      await hydrate(true)
    } else {
      awaitingNext.value = false
      roundResult.value = null
      nextRoundNo.value = null
      pushLog('继续下一局')
    }
  } catch (e) {
    errorMsg.value = String(e)
  } finally {
    busy.value = false
  }
}

async function newGame() {
  clearLocal()
  pushLog('已新开一场')
  try {
    await fetch('/api/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rounds: state.totalRounds || 8 }),
    })
  } catch (e) {
    errorMsg.value = String(e)
  }
}

function clearLocal() {
  state.rounds = []
  state.required = null
  state.finished = false
  state.view = null
  state.cumulative = [0, 0, 0, 0]
  errorMsg.value = ''
  suggestion.value = null
  awaitingNext.value = false
  roundResult.value = null
  nextRoundNo.value = null
  handOrder.value = []
  dragOrder.value = null
}

// 新的一局 / 新一场开始时，重置为自动排序
watch(
  () => state.view?.round_no,
  () => {
    if (state.view?.my_hand) resetHandOrder(state.view.my_hand)
  },
)

// 窗口尺寸变化 / 内容高度变化 → 重新计算缩放
let ro = null
function onResize() {
  recomputeZoom()
}

onMounted(() => {
  recomputeZoom()
  window.addEventListener('resize', onResize)
  if (typeof ResizeObserver !== 'undefined' && stageRef.value) {
    ro = new ResizeObserver(() => recomputeZoom())
    ro.observe(stageRef.value)
  }
})
onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  ro && ro.disconnect()
  es && es.close()
})

// --------------------------------------------------------------------------- //
// SSE
// --------------------------------------------------------------------------- //
let es = null
let watchdog = null
onMounted(() => {
  hydrate()
  es = new EventSource('/api/stream')
  es.onmessage = (ev) => {
    let msg
    try {
      msg = JSON.parse(ev.data)
    } catch {
      return
    }
    if (msg.type === 'state') {
      state.status = 'running'
      state.view = msg.view // 整体替换：关天眼时后端不下发 others_hands，合并会残留
      if (msg.view?.my_hand) handleHand(msg.view.my_hand)
      // 服务端下发的「当前待决策点」为权威：据此同步，
      // 即使 SSE 丢事件也能自愈（不会卡在旧按钮/缺按钮）。
      if ('required' in msg) {
        state.required = msg.required
        if (msg.required?.view) state.view = msg.required.view
      }
    } else if (msg.type === 'action_required') {
      state.view = msg.view
      state.required = msg
      selected.value = null
      // 新决策点：先清空上一个建议，等后端 suggestion 事件到来（显示「计算中…」）
      suggestion.value = null
      if (msg.view?.my_hand) handleHand(msg.view.my_hand)
      pushLog(phaseLabel(msg.phase) + ' · ' + msg.actions.map((a) => a.kind).join('/'))
      if (msg.phase === 'draw' && msg.view?.drawn_tile) {
        pushLog('🀄 摸到 ' + tileLabel(msg.view.drawn_tile))
      }
    } else if (msg.type === 'suggestion') {
      // 引擎在人类决策点上的推荐（仅供你观察，不代打）
      suggestion.value = msg
    } else if (msg.type === 'new_game') {
      clearLocal()
      pushLog('新的一局已开始')
    } else if (msg.type === 'round_end') {
      state.rounds.push(msg.result)
      state.cumulative = msg.cumulative
      const r = msg.result
      // 每局结束：把结果设为当前展示局面，并弹出结果面板（含「开下一局」）
      roundResult.value = r
      awaitingNext.value = !!msg.awaiting_next
      nextRoundNo.value = msg.next_round_no ?? null
      state.required = null
      suggestion.value = null
      pushLog(
        r.is_flow
          ? `第 ${r.round_no} 局 流局`
          : `第 ${r.round_no} 局 ${relNames.value[r.winner] || r.winner} 胡 ${(r.detail || []).join('+')} (${r.fan}番)`,
      )
    } else if (msg.type === 'match_end') {
      state.finished = true
      state.required = null
      awaitingNext.value = false
      nextRoundNo.value = null
      pushLog('本场结束')
    } else if (msg.type === 'error') {
      errorMsg.value = msg.message
    }
  }
  es.onerror = () => {
    state.status = 'reconnecting'
  }
})
onUnmounted(() => es && es.close())

// 看门狗：本地无「等待决策」且未在提交时，周期性向服务端对账。
// 若服务端其实有决策点（例如 SSE 丢事件、连接重建），自动补上——
// 相当于替你自动刷新，消除「必须手动刷新才能继续」的卡死。
onMounted(() => {
  watchdog = setInterval(async () => {
    if (busy.value || state.required || state.finished) return
    try {
      const r = await fetch('/api/state')
      const j = await r.json()
      if (busy.value || state.required) return
      state.cumulative = j.cumulative || state.cumulative
      state.rounds = j.rounds || state.rounds
      if (j.required) {
        state.required = j.required
        if (j.required.view?.my_hand) handleHand(j.required.view.my_hand)
        state.view = j.required.view
        pushLog('已自动同步决策点')
      } else if (j.state?.phase === 'draw' || j.state?.phase?.startsWith?.('response')) {
        state.view = j.state
      }
    } catch {
      /* 网络抖动忽略，下轮再试 */
    }
  }, 1500)
})
onUnmounted(() => {
  clearInterval(watchdog)
  es && es.close()
})
</script>

<template>
  <div class="app">
    <div class="stagebox" :style="boxStyle">
    <div class="stage" ref="stageRef" :style="stageStyle">
    <header>
      <h1>杭州麻将 <span class="vs">你 vs 3×{{ decider }}</span></h1>
      <div class="hud">
        <span>第 {{ state.view?.round_no || 1 }}/{{ state.totalRounds }} 局</span>
        <span>牌墙余 {{ state.view?.wall_remaining ?? '—' }}</span>
        <span v-if="state.view?.god">
          手留白 {{ state.view.god.hand_gods }} · 链 {{ state.view.god.chain_count }} · 飘 {{ state.view.god.piao_count }}
        </span>
        <span v-if="state.view?.catch_play" class="warn">抓打圈</span>
        <span class="zoomctl">
          <button class="zoombtn" @click="zoomOut" title="缩小">－</button>
          <span class="zoomval">{{ Math.round(zoom * 100) }}%</span>
          <button class="zoombtn" @click="zoomIn" title="放大">＋</button>
        </span>
        <button class="reveal" :class="{ on: reveal }" @click="toggleReveal">
          {{ reveal ? '👁 天眼：开' : '👁 开天眼' }}
        </button>
        <button class="reveal" :class="{ on: suggestEnabled }" @click="toggleSuggest">
          {{ suggestEnabled ? '🧠 建议：开' : '🧠 建议：关' }}
        </button>
        <button class="report" @click="openReport">📸 报障</button>
        <button class="newgame" @click="newGame">新开一场</button>
      </div>
    </header>

    <main class="board">
      <!-- 对家（上） -->
      <div class="area north" v-if="seatsByPos.top !== null">
        <SeatCard v-bind="seatData(seatsByPos.top)" />
      </div>

      <!-- 上家（左） / 下家（右） -->
      <div class="area west" v-if="seatsByPos.left !== null">
        <SeatCard v-bind="seatData(seatsByPos.left)" vertical />
      </div>
      <div class="area east" v-if="seatsByPos.right !== null">
        <SeatCard v-bind="seatData(seatsByPos.right)" vertical />
      </div>

      <!-- 自己（下）：只显示名牌与分数；手牌在下方大字区 -->
      <div class="area south" v-if="seatsByPos.bottom !== null">
        <SeatCard v-bind="seatData(seatsByPos.bottom)" hide-melds />
      </div>

      <!-- 中央牌池：四方弃牌围成一圈，中间放对局信息 -->
      <div class="pool">
        <div class="ptop"><Discards :tiles="poolOf(seatsByPos.top)" /></div>
        <div class="pleft"><Discards :tiles="poolOf(seatsByPos.left)" vertical /></div>
        <div class="pcenter">
          <div class="pr-wind">第 {{ state.view?.round_no || 1 }} 局</div>
          <div class="pr-wall">余 {{ state.view?.wall_remaining ?? '—' }}</div>
          <div class="phase" v-if="state.required">{{ phaseLabel(state.required.phase) }}</div>
          <div class="phase idle" v-else-if="!state.finished">等待…</div>
          <div class="phase over" v-else>结束</div>
        </div>
        <div class="pright"><Discards :tiles="poolOf(seatsByPos.right)" vertical /></div>
        <div class="pbottom"><Discards :tiles="poolOf(seatsByPos.bottom)" /></div>
      </div>
    </main>

    <!-- 我的牌区：副露在手牌右侧（主流做法），动作按钮在手牌上方 -->
    <section class="mine">
      <!-- 本局结果面板：每局结束弹出，手动点「开下一局」才继续 -->
      <div class="roundres" v-if="roundResult && !state.finished">
        <div class="rr-head">
          <span class="rr-title">第 {{ roundResult.round_no }} 局结束</span>
          <span class="rr-outcome" :class="{ flow: roundResult.is_flow }">
            {{ roundResult.is_flow ? '流局' : ((relNames[roundResult.winner] || roundResult.winner) + ' 胡牌') }}
          </span>
        </div>
        <div class="rr-body" v-if="!roundResult.is_flow">
          <span class="rr-label">牌型</span>
          <b class="rr-tile" v-for="(c, i) in (roundResult.detail || [])" :key="i">{{ tileLabel(c) }}</b>
          <span class="rr-fan">{{ roundResult.fan }} 番</span>
        </div>
        <div class="rr-scores">
          <span v-for="(s, i) in (roundResult.scores || [])" :key="i" class="rr-sc" :class="{ me: i === state.humanSeat }">
            {{ relNames[i] || i }} <b :class="s > 0 ? 'pos' : (s < 0 ? 'neg' : '')">{{ s > 0 ? '+' : '' }}{{ s }}</b>
          </span>
        </div>
        <button class="rr-next" :disabled="!awaitingNext || busy" @click="startNextRound">
          {{ awaitingNext ? ('开下一局 ▶' + (nextRoundNo ? '（第 ' + nextRoundNo + ' 局）' : '')) : '进行中…' }}
        </button>
        <span class="rr-hint" v-if="awaitingNext">看完本局结果，点右侧按钮继续</span>
      </div>
      <!-- 自摸成胡：强烈提醒 -->
      <div class="win" v-if="canHu">
        <span class="w-tag">可以胡了</span>
        <span class="w-text"><b>自摸胡</b> — 点「胡」结束本局</span>
      </div>
      <!-- 响应提示：别人打出的牌，我能碰/杠/吃时强烈提醒 -->
      <div class="respond" v-if="isResponse">
        <span class="r-tag">轮到你决策</span>
        <span class="r-text">
          {{ offeredBy ? offeredBy + ' 打出' : '有人打出' }}
          <b class="r-tile">{{ tileLabel(offeredTile) }}</b>
          <template v-if="isChi && chiCombos.length">
            — 你可以<b>吃</b>，用
            <b v-for="(c, i) in chiCombos" :key="i" class="r-combo">{{ c.map(tileLabel).join('+') }}</b>
          </template>
          <template v-else>
            — 你可以 <b>{{ responseVerb }}</b>
          </template>
        </span>
      </div>
      <!-- 摸牌提示：本回合摸到的牌 -->
      <div class="draw" v-if="drawnTile">
        <span class="d-tag">摸牌</span>
        <span class="d-text">你摸到了 <b class="d-tile">{{ tileLabel(drawnTile) }}</b></span>
      </div>
      <!-- 建议：用当前决策档在本决策点的推荐，供你观察其出牌逻辑（不代打） -->
      <div class="suggest" v-if="suggestEnabled && state.required && !state.finished">
        <span class="s-tag">{{ decider }} 建议</span>
        <span class="s-text">{{ suggestText }}</span>
        <span class="s-meta" v-if="suggestion && suggestion.elapsed_ms != null">{{ Math.round(suggestion.elapsed_ms) }}ms</span>
      </div>
      <div class="actions">
        <button
          v-if="canDiscard"
          class="act discard"
          :disabled="busy || !selected || !canDiscardCode(selected)"
          @click="confirmDiscard"
        >
          出牌<span v-if="selected" class="hint">{{ selected }}</span>
        </button>
        <button
          v-for="(a, i) in otherActions"
          :key="i"
          class="act"
          :class="[a.kind, { pulse: a.kind !== 'pass' && (isResponse || canHu), suggested: isSuggested(a) }]"
          :disabled="busy"
          @click="doAction(a)"
        >
          {{ actionLabel(a) }}
          <span v-if="actionHint(a)" class="hint">{{ actionHint(a) }}</span>
        </button>
        <span class="tip" v-if="canDiscard">点选要出的牌，再按「出牌」；直接拖动牌可调整顺序</span>
      </div>
      <div class="handrow">
        <div class="myhand">
          <Tile
            v-for="(item, idx) in myHandKeyed"
            :key="item.key"
            :code="item.code"
            :selectable="canDiscardCode(item.code)"
            :selected="selected === item.code"
            :drawn="drawnIndex >= 0 && idx === drawnIndex"
            :suggested="item.code === suggestDiscardCode"
            draggable="true"
            @click="onClickTile(item.code)"
            @dragstart="onDragStart(item.code, idx, $event)"
            @dragover="onDragOver(idx, $event)"
            @dragend="onDragEnd"
            @drop.prevent="onDragEnd"
          />
        </div>
        <div class="my-melds" v-if="myMelds.length">
          <span class="mm-label">副露</span>
          <Melds :melds="myMelds" />
        </div>
      </div>
      <div class="err" v-if="errorMsg">{{ errorMsg }}</div>
    </section>
    </div><!-- /stage -->
    </div><!-- /stagebox -->

    <aside class="log">
      <h2>对局记录</h2>
      <ul>
        <li v-for="(l, i) in state.log" :key="i">{{ l }}</li>
      </ul>
    </aside>

    <!-- 报障弹窗：写下想法 → 确认，当前牌局 + 引擎建议一起存档，供离线复现 -->
    <div class="modal-mask" v-if="reportOpen" @click.self="closeReport">
      <div class="modal">
        <h3>📸 报障 · 保存当前场景</h3>
        <p class="modal-sub">保存后你<strong>不用再保持这个场景</strong>：快照含完整牌局与引擎建议，我可离线精确复现并查原因。</p>
        <div class="modal-snap" v-if="state.required">
          <span>当前决策点：<b>{{ phaseLabel(state.required.phase) }}</b></span>
          <span v-if="offeredTile">被响应：<b>{{ tileLabel(offeredTile) }}</b></span>
          <div v-if="suggestText" class="modal-sug">{{ suggestText }}</div>
        </div>
        <div class="modal-snap warn" v-else>当前没有待决策点（会保存现状 + 最近的局面）</div>
        <textarea v-model="reportText" rows="4"
          placeholder="写下你的想法：你觉得该怎么打、为什么…（可留空）"></textarea>
        <div class="modal-actions">
          <span class="modal-msg" v-if="reportDone">✅ 已保存 {{ reportDone }}</span>
          <span class="modal-msg err" v-else-if="reportError">{{ reportError }}</span>
          <button class="modal-cancel" @click="closeReport">取消</button>
          <button class="modal-ok" :disabled="reportBusy" @click="submitReport">确认报障</button>
        </div>
      </div>
    </div>
  </div>
</template>
