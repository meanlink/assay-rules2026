import {esc,fmt} from '../../utils/dom.js';
const BADGE_CLASS={
 '复验通过':'badge-pass',
 '复验样本不足':'badge-warn',
 '复验不支持':'badge-fail',
 '复验未观测':'badge-muted'
};
export function table(columns,rows,empty='暂无记录。请先导入业务数据，再创建分析任务。'){
 if(!rows.length)return '<div class="empty">'+esc(empty)+'</div>';
 return '<div class="table-wrap"><table><thead><tr>'+columns.map(c=>'<th>'+esc(c.label)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+columns.map(c=>{
   const raw=r[c.key];
   if(c.badge){const cls=BADGE_CLASS[raw]||'badge-muted';return '<td><span class="badge '+cls+'">'+esc(fmt(raw))+'</span></td>';}
   return '<td>'+esc(fmt(raw))+'</td>';
 }).join('')+'</tr>').join('')+'</tbody></table></div>';
}
