'use strict';
const data = window.MODULE_CATALOG;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statusLabel = {expanded:'已验证转写', stateless:'仅训练无状态', unsupported:'未实现', structural:'结构 / 抽象类'};
const dot = {expanded:'verified',stateless:'partial',unsupported:'missing',structural:''};
const color = op => ['REPEAT','SCAN'].includes(op)?'#7854b5':['PROJECT','MIX','COMPARE'].includes(op)?'#5366db':['REDUCE','NORM','NORMALIZE','AGGREGATE'].includes(op)?'#b26a14':['ROUTE','RESHAPE','PERMUTE','BROADCAST','SHIFT','CONCAT'].includes(op)?'#168173':'#b34d75';
const fmt = n => typeof n === 'number' ? n.toExponential(3) : '未记录';
const publicRecords = data.records.filter(r => !r.case.includes(':'));
const count = status => publicRecords.filter(r => r.status === status).length;
$('#coverage-summary').innerHTML = `<div class="summary-item"><strong>${publicRecords.length}</strong>公开类</div><div class="summary-item"><span class="dot verified"></span><strong>${count('expanded')}</strong>已转写</div><div class="summary-item"><span class="dot missing"></span><strong>${count('unsupported')}</strong>未实现</div><div class="summary-item"><strong>${count('structural')}</strong>结构类</div><div class="summary-item"><strong>${data.records.length-publicRecords.length}</strong>变体</div><span class="coverage-stamp">PYTORCH ${esc(data.torch)} · ${data.operations.length} PRIMITIVES</span>`;
const date = v => v ? new Date(v).toLocaleString('zh-CN',{hour12:false}) : '未记录';
$('#catalog-provenance').textContent = `源码同步：${date(data.generatedAt)} · 展开代码验证：${date(data.validatedAt)} · CPU / float64。原语徽标表示静态调用次数，并非运行时展开次数或完整依赖图。`;
function details(r) {
  const methods=(r.methods||[]).filter(m=>m.operations.length).map(m=>`<span class="call-method"><b>${esc(m.name)}</b>：${esc(m.operations.join(' · '))}</span>`).join('');
  return `<details><summary>展开配置、输入输出与验证记录</summary><dl><dt>参考配置</dt><dd class="mono">${esc(r.sourceRepr || r.config || '不适用')}</dd><dt>输入契约</dt><dd class="mono">${esc(r.inputSignature || '未记录可执行输入契约')}</dd><dt>参考输出</dt><dd class="mono">${esc(r.outputs.length?r.outputs.map(o=>JSON.stringify(o.shape)+' · '+o.dtype).join('\n'):'无数值输出')}</dd><dt>前向对比</dt><dd>${r.validation==='equivalent'||r.validation==='partial_state'?`最大绝对误差 ${fmt(r.maxAbsError)}`:esc(statusLabel[r.status])}</dd><dt>反向对比</dt><dd>${r.backward?`${esc(r.backward.status)} · 最大绝对误差 ${fmt(r.backward.max_abs_error)} · ${r.backward.checked_tensors||0} 个张量`:'不适用 / 未记录'}</dd><dt>适用边界</dt><dd>${esc(r.limitation||'仅针对列出的配置验证，不代表完整 PyTorch API 兼容。')}${r.status==='stateless'?'<br>数值对比不包含完整的训练状态更新。':''}</dd><dt>文件摘要</dt><dd class="mono">SHA256 ${esc(r.sha256)}</dd></dl>${methods?`<p>按源码位置列出的原语调用（含独立循环体，不表示线性依赖）：${methods}</p>`:''}</details>`;
}
function render() {
  const query=$('#module-query').value.trim().toLowerCase();
  const status=$('#module-status').value;
  const variants=$('#include-variants').checked;
  const records=data.records.filter(r=>(variants||!r.case.includes(':'))&&(status==='all'||r.status===status)&&`${r.name} torch.nn.${r.name} ${r.case} ${Object.keys(r.operations).join(' ')}`.toLowerCase().includes(query));
  $('#visible-count').textContent=`${records.length} / ${variants?data.records.length:publicRecords.length} 条定义`;
  $('#catalog-empty').hidden=records.length>0;
  $('#replacement-rows').innerHTML=records.map(r=>{
    const route=encodeURIComponent(r.case);
    const operations=Object.entries(r.operations);
    const implemented=r.status==='expanded'||r.status==='stateless';
    const replacement=operations.length?`<div class="op-badges">${operations.map(([op,n])=>`<a class="op-badge" style="--op:${color(op)}" href="primitives.html#${op}">${['REPEAT','SCAN'].includes(op)?'↻ ':''}${op} ×${n}</a>`).join('')}</div>`:`<p class="replacement-reason">${r.status==='structural'?'属于宿主结构或参数容器，目前无独立数值平替。':implemented?'恒等映射：直接返回输入，不需要数值原语。':'目前没有可执行的原语平替；源码明确抛出 NotImplementedError。'}</p>`;
    return `<tr data-module="${esc(r.case)}"><td><a class="replacement-name" href="index.html#${route}">${r.case.includes(':')?'':'nn.'}${esc(r.name)}</a>${r.case.includes(':')?`<small class="variant-name">${esc(r.case.split(':')[1])}</small>`:''}</td><td><span class="module-status ${r.status==='expanded'?'':'warn'}"><span class="dot ${dot[r.status]}"></span>${statusLabel[r.status]}</span></td><td>${replacement}${details(r)}</td><td><div class="replacement-code-links"><a href="index.html#${route}">${implemented?'查看原语 forward':'查看当前定义'} ↗</a>${r.definition?`<a href="index.html?view=definition#${route}">ParameterDefinition ↗</a>`:''}<a href="code/${encodeURI(r.file)}" download>下载 Python ↓</a></div></td></tr>`;
  }).join('');
}
$('#module-query').addEventListener('input',render);
$('#module-status').addEventListener('change',()=>{if($('#module-status').value==='stateless')$('#include-variants').checked=true;render();});
$('#include-variants').addEventListener('change',render);
render();
