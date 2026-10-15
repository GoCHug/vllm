<template>
<header>
  <div class="wrap">
    <h1>vLLM V1 KVCache 实验数据可视化</h1>
    <div class="sub" id="hdrSub">正在加载 manifest.json …</div>
  </div>
</header>
<nav><div class="wrap" id="navTabs"></div></nav>
<main><div class="wrap">
  <div id="errbox"></div>
  <section class="view" id="view-overview"></section>
  <section class="view" id="view-kvc"></section>
  <section class="view" id="view-p1d"></section>
  <section class="view" id="view-matrix"></section>
  <section class="view" id="view-flow"></section>
  <section class="view" id="view-sandbox"></section>
  <section class="view" id="view-browser"></section>
</div></main>
<div id="heatTip"></div>
</template>

<script setup>
import { onMounted } from 'vue'
import { boot } from './lib/legacy.js'
// Vue 渲染静态骨架; 全部渲染/交互逻辑走 legacy 桥接层(自原 viz.html JS 搬运+升级)
onMounted(() => { boot() })
</script>
