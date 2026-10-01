<!-- 导航栏上的"目录"按钮：点开后列出所有页，点击跳转 -->
<script setup lang="ts">
import { computed, ref } from 'vue'
import { useNav } from '@slidev/client'

const { slides, currentPage, go } = useNav()
const open = ref(false)

const items = computed(() =>
  slides.value.map(route => ({
    no: route.no,
    title: route.meta?.slide?.title || `第 ${route.no} 页`,
    isSection: route.meta?.slide?.frontmatter?.layout === 'section',
  })),
)

function jump(no: number) {
  go(no)
  open.value = false
}
</script>

<template>
  <button class="slidev-icon-btn" title="目录" @click="open = !open">
    <div class="i-carbon:list" />
  </button>

  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-100" @click="open = false">
      <div
        class="toc-panel fixed right-4 top-4 bottom-16 w-90 overflow-y-auto rounded-lg shadow-xl p-4 text-sm"
        @click.stop
      >
        <div class="flex items-center justify-between mb-3">
          <span class="font-bold text-base">目录</span>
          <button class="slidev-icon-btn" title="关闭" @click="open = false">
            <div class="i-carbon:close" />
          </button>
        </div>
        <div
          v-for="item in items"
          :key="item.no"
          class="toc-item flex gap-2 px-2 py-1 rounded cursor-pointer"
          :class="{
            'font-bold mt-3': item.isSection,
            'pl-5': !item.isSection,
            'toc-current': item.no === currentPage,
          }"
          @click="jump(item.no)"
        >
          <span class="opacity-50 w-6 text-right tabular-nums shrink-0">{{ item.no }}</span>
          <span>{{ item.title }}</span>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.toc-panel {
  background: #ffffff;
  color: #1f2937;
  border: 1px solid #e5e7eb;
}
.toc-item:hover {
  background: rgba(14, 165, 233, 0.12);
}
.toc-current {
  background: rgba(14, 165, 233, 0.22);
}
:global(html.dark) .toc-panel {
  background: #111827;
  color: #e5e7eb;
  border-color: #374151;
}
</style>
