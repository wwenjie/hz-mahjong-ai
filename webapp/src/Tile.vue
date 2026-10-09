<script setup>
import { computed } from 'vue'
import { tileInfo, isGod } from './tiles.js'

const props = defineProps({
  code: { type: String, default: '' },
  selectable: { type: Boolean, default: false },
  selected: { type: Boolean, default: false },
  small: { type: Boolean, default: false },
  mini: { type: Boolean, default: false },
  ghost: { type: Boolean, default: false },
  drawn: { type: Boolean, default: false },
  suggested: { type: Boolean, default: false },
})

const info = computed(() => tileInfo(props.code))
const god = computed(() => isGod(props.code))
</script>

<template>
  <div class="tile" :class="{ selectable, selected, small, mini, ghost, god, drawn, suggested }">
    <template v-if="info.honor">
      <span class="honor">{{ info.suit }}</span>
    </template>
    <template v-else>
      <span class="num">{{ info.num }}</span>
      <span class="suit">{{ info.suit }}</span>
    </template>
    <span v-if="god && !mini" class="god-badge">财神</span>
  </div>
</template>
