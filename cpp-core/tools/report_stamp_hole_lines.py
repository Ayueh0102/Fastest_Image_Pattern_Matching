"""Render the stamp-hole pilot's measured curves and per-line review, offline."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_rows(path):
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    for row in rows:
        for key,value in list(row.items()):
            if key.endswith('_px'):
                row[key] = float(value) if value else None
            elif key.endswith('_index'):
                row[key] = int(value) if value else None
            elif key.startswith('accepted_') or key=='reference_pair_reliable':
                row[key] = value == 'True'
    return rows


def plot_svg(rows, width=880, height=205):
    lo,hi=min(r['pitch_index'] for r in rows),max(r['pitch_index'] for r in rows)
    x=lambda k: 44+(k-lo)/(hi-lo)*(width-64)
    y=lambda d: 28+(8-d)/16*(height-58)
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="齒孔偏移曲線，單位像素">']
    for v in [-5,0,5]:
        s.append(f'<path d="M44 {y(v):.1f} H{width-20}" stroke="{"#adbac1" if v==0 else "#dec6c6"}" stroke-dasharray="5 4"/><text x="5" y="{y(v)+4:.1f}" font-size="12" fill="#49606b">{v:+d}</text>')
    for k in np.linspace(lo,hi,9):
        s.append(f'<text x="{x(k):.1f}" y="{height-8}" text-anchor="middle" font-size="12" fill="#49606b">{k:.0f}</text>')
    for field,base_color,radius in [('good_normal_to_curve_px','#77a89b',2),('bad_normal_to_curve_px','#1a688e',2.4)]:
        for row in rows:
            value=row[field]
            if value is None:continue
            color='#cf343c' if field.startswith('bad') and abs(value)>=5 else base_color
            s.append(f'<circle cx="{x(row["pitch_index"]):.2f}" cy="{y(value):.2f}" r="{radius}" fill="{color}"><title>{row["line"]}:{row["pitch_index"]} {field} = {value:.2f} px</title></circle>')
    s.append('</svg>')
    return ''.join(s)


def render_annotated_pair(directory, result, rows, ignore_offset_at_most=1.0, minimum_offset=None):
    """Label ALL measured holes above threshold independently on both photographs.

    Labels are distances to each photograph's OWN curve, not bad-minus-good
    displacement. Shared intersections receive the largest normal offset label;
    every above-threshold line measurement remains in the audit CSV.
    """
    from stamp_hole_inspect import distance_to_curve

    threshold = 5.0
    font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    font = lambda size: ImageFont.truetype(font_path, size)
    blue, green, red = '#0074c8', '#009a69', '#d52c36'
    manifest = []
    visible_counts = []
    coverage = []
    pending_manifest = []
    floor = ignore_offset_at_most if minimum_offset is None else minimum_offset
    inclusive = minimum_offset is not None
    relation, ignored_relation = ('≥', '<') if inclusive else ('>', '≤')
    suffix = f'-all-{"ge" if inclusive else "gt"}{floor:g}px'
    qualifies = lambda value: abs(value) >= floor if inclusive else abs(value) > floor

    def direction(axis, value):
        word = ('下' if value > 0 else '上') if axis == 'H' else ('右' if value > 0 else '左')
        # Do not round a value just above the threshold into an apparently ignored value.
        amount = f'>{floor:g}' if not inclusive and round(abs(value), 1) <= floor else f'{abs(value):.1f}'
        return f'{word}{amount}'

    def overlap(a, b):
        return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))

    for image_index, name in enumerate(['bad', 'good']):
        source = Image.open(result['input_images'][image_index]).convert('RGB')
        width, height = source.size
        canvas = source.copy()
        draw = ImageDraw.Draw(canvas)
        data = np.load(directory / f'detection{image_index+1}.npz')
        obs, accepted = data['observations'], data['accepted']
        valid_rows = [r for r in rows if r[f'{name}_normal_to_curve_px'] is not None]
        measurable = {r['site_index'] for r in valid_rows}
        selected_rows = [r for r in valid_rows if qualifies(r[f'{name}_normal_to_curve_px'])]
        by_site = {}
        for row in selected_rows:
            old = by_site.get(row['site_index'])
            if old is None or abs(row[f'{name}_normal_to_curve_px']) > abs(old[f'{name}_normal_to_curve_px']):
                by_site[row['site_index']] = row
        highlights = sorted(by_site.values(),key=lambda r:-abs(r[f'{name}_normal_to_curve_px']))
        for cx, cy in obs[sorted(measurable), :2]:
            draw.ellipse((cx-9, cy-9, cx+9, cy+9), outline=green, width=2)
        pending = sorted(set(range(len(obs))) - measurable)
        for cx, cy in data['prediction'][pending]:
            draw.rectangle((cx-7,cy-7,cx+7,cy+7),outline='#7a6d86',width=2)
        for index in pending:
            pending_manifest.append(dict(image=image_index+1,site_index=index,
                nominal_line=str(data['ids'][index,0])+str(int(data['ids'][index,1])+1),
                pitch_index=int(data['ids'][index,2]),predicted_x_px=float(data['prediction'][index,0]),
                predicted_y_px=float(data['prediction'][index,1]),native_accepted=bool(accepted[index]),
                reason='insufficient_geometry_support' if accepted[index] else 'not_reliably_detected'))

        reserved_labels = []
        for line, model in result['curves'].items():
            axis = 0 if line.startswith('H') else 1
            supported = [r for r in rows if r['line'] == line and r[f'accepted_{name}']]
            indices = [r['site_index'] for r in supported]
            points = obs[indices, :2]
            u = np.linspace(points[:, axis].min(), points[:, axis].max(), 900)
            curve = np.polynomial.Polynomial(model[f'{name}_coeff'])
            coords = np.column_stack([u, curve(u)] if axis == 0 else [curve(u), u])
            draw.line([tuple(p) for p in coords], fill=blue, width=3)
            x, y = coords[0]
            lx, ly = (x-102, y-23) if axis == 0 else (x-28, y-56)
            box = (lx-7, ly-3, lx+73, ly+42)
            reserved_labels.append(box)
            draw.rounded_rectangle(box, radius=6, fill='#eef7ff')
            draw.text((lx, ly), line, font=font(36), fill=blue)

        measurements = []
        for row in highlights:
            line = row['line']
            i = row['site_index']
            center = obs[i, :2]
            curve = np.polynomial.Polynomial(result['curves'][line][f'{name}_coeff'])
            distance, foot, _ = distance_to_curve(center[None, :], curve, 0 if line[0] == 'H' else 1)
            value = float(distance[0])
            assert abs(value-row[f'{name}_normal_to_curve_px']) < 1e-7
            measurements.append((i, row, center, foot[0], value))
        for row in selected_rows:
            i = row['site_index'];value = row[f'{name}_normal_to_curve_px']
            manifest.append(dict(image=image_index+1,site_index=i,line=row['line'],
                                 pitch_index=row['pitch_index'],x_px=float(obs[i,0]),y_px=float(obs[i,1]),
                                 signed_normal_px=value,label_text=direction(row['line'][0],value),
                                 label_visible=by_site[i] is row,ignore_offset_at_most_px=None if inclusive else floor,
                                 minimum_offset_px=minimum_offset,
                                 reference_pair_reliable=row['reference_pair_reliable']))
        visible_counts.append(len(measurements))
        displayed_ids = {m[0] for m in measurements}
        unmarked = set(by_site) - displayed_ids
        assert not unmarked, f'Above-threshold holes omitted from annotation: {sorted(unmarked)}'
        coverage.append(dict(image=image_index+1,source=name,threshold_px=floor,threshold_inclusive=inclusive,nominal_sites=len(obs),
                             native_accepted=int(accepted.sum()),geometry_measurable=len(measurable),
                             no_reliable_measurement=len(pending),
                             native_rejected=int((~accepted).sum()),
                             accepted_without_geometry=sum(bool(accepted[i]) for i in pending),
                             all_above_threshold_holes=len(by_site),above_threshold_line_measurements=len(selected_rows),
                             displayed_holes=len(measurements),unmarked_above_threshold_holes=len(unmarked),
                             beyond_5px_holes=sum(abs(m[4])>=5 for m in measurements),
                             pending_site_indices=pending))

        # Greedy placement keeps each numeric label separate. Leaders point to the
        # original-coordinate hole; label positions do not affect measurements.
        occupied = list(reserved_labels)
        label_boxes = []
        centers = np.array([m[2] for m in measurements]).reshape(-1,2)
        for number, row, center, foot, value in measurements:
            x, y = map(float, center)
            text = direction(row['line'][0],value)
            bbox = draw.textbbox((0, 0), text, font=font(24))
            bw, bh = bbox[2]-bbox[0]+12, 34
            choices = []
            occupied_array=np.array(occupied).reshape(-1,4)
            for dy in [-52, 24, -94, 66, -136, 108, -178, 150, -220, 192, -262, 234]:
                for dx in [-bw/2, 24, -bw-24, 95, -bw-95, 166, -bw-166]:
                    left = max(30, min(width-bw-30, x+dx))
                    top = max(460, min(3570-bh, y+dy))
                    box = (left, top, left+bw, top+bh)
                    cross_width=np.maximum(0,np.minimum(box[2],occupied_array[:,2])-np.maximum(box[0],occupied_array[:,0]))
                    cross_height=np.maximum(0,np.minimum(box[3],occupied_array[:,3])-np.maximum(box[1],occupied_array[:,1]))
                    penalty=float(np.sum(cross_width*cross_height))*100000
                    penalty += np.sum((centers[:,0]>=left-9)&(centers[:,0]<=left+bw+9)&
                                      (centers[:,1]>=top-9)&(centers[:,1]<=top+bh+9))*1000
                    penalty += np.hypot(left+bw/2-x, top+bh/2-y)
                    choices.append((penalty, box))
            _, box = min(choices, key=lambda item:item[0])
            occupied.append(box)
            color = red if abs(value)>=5 else ('#b35300' if abs(value)>=3 else '#9a6800')
            lx = min(max(x, box[0]), box[2]);ly = min(max(y, box[1]), box[3])
            draw.line([(x, y), (lx, ly)], fill='white', width=3)
            draw.line([(x, y), (lx, ly)], fill=color, width=1)
            fx, fy = foot
            draw.line([(fx-5, fy), (fx+5, fy)], fill=blue, width=2)
            draw.line([(fx, fy-5), (fx, fy+5)], fill=blue, width=2)
            draw.line([(fx, fy), (x, y)], fill=color, width=3)
            radius=16 if abs(value)>=5 else 11
            draw.ellipse((x-radius-1,y-radius-1,x+radius+1,y+radius+1),outline='white',width=4)
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
            draw.rounded_rectangle(box,radius=5,fill='white',outline=color,width=1)
            draw.text((box[0]+6,box[1]+5),text,font=font(24),fill=color)
            label_boxes.append((box,text,color))

        # Later leaders can cross an earlier label. Paint all text boxes last so
        # dense 1–5 px annotations remain readable on the original-resolution image.
        for box,text,color in label_boxes:
            draw.rounded_rectangle(box,radius=5,fill='white',outline=color,width=1)
            draw.text((box[0]+6,box[1]+5),text,font=font(24),fill=color)

        maximum = max(abs(r[f'{name}_normal_to_curve_px']) for r in rows
                      if r[f'{name}_normal_to_curve_px'] is not None)
        title = f'圖{image_index+1}｜{"壞品" if name=="bad" else "良品"}：全部 {relation} {floor:g} px 孔位，逐孔標註'
        draw.rounded_rectangle((55, 112, width-55, 355), radius=16, fill='#f4f8fc')
        draw.text((90, 128), title, font=font(62), fill='#163445')
        count_text=f'已標 {len(measurements)} 個 {relation} {floor:g} px 孔位（其中 ≥ 5 px：{coverage[-1]["beyond_5px_holes"]} 個）；這是距線殘差，不是已確認缺陷數'
        draw.text((90,212),f'可量測 {len(measurable):,} 個   ｜   未可靠量測名義孔位 {len(pending)} 個（灰框）   ｜   H1–H6 ＋ V1–V11',font=font(39),fill='#254e63')
        draw.text((90,276),count_text,font=font(39),fill='#754b05')
        draw.rounded_rectangle((55, 3730, width-55, height-50), radius=18, fill='#f4f8fc')
        draw.text((90,3754),f'綠圈＝可量測孔位   藍線＝虛擬格線   橙圈＝{relation} {floor:g} 且 < 5 px   紅圈＝≥ 5 px   灰框＝尚無可靠量測',font=font(39),fill='#254e63')
        draw.text((90, 3821), '數值＝孔心到「該張圖」虛擬格線的最短距離；上／下／左／右表示偏移方向。', font=font(39), fill='#254e63')
        draw.text((90, 3888), f'原圖 {width} × {height}；單位 px（像素），不是毫米。可量區最大距線偏移 {maximum:.1f} px。', font=font(39), fill='#254e63')
        draw.text((90,3955),f'{ignored_relation} {floor:g} px 不顯示數值；交點標較大偏移。灰框不等於缺孔；底邊遮擋區不延伸格線。請放大原圖閱讀。',font=font(36),fill='#536f7c')
        output = directory / f'{image_index+1:02d}-{name}-holes-grid-offsets{suffix}.png'
        canvas.save(output)
        canvas.resize((1800, round(height*1800/width)), Image.Resampling.LANCZOS).save(
            directory / f'{image_index+1:02d}-{name}-holes-grid-offsets{suffix}-preview.jpg', quality=95)
        if any(overlap(a,b)>0 for i,a in enumerate(occupied) for b in occupied[i+1:]):
            raise AssertionError('Offset label boxes overlap; adjust layout before delivery')
    with (directory/f'annotated-pair-labels{suffix}.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        fields = list(manifest[0]) if manifest else ['image','site_index','line','pitch_index','x_px','y_px',
                 'signed_normal_px','label_text','label_visible','ignore_offset_at_most_px','minimum_offset_px','reference_pair_reliable']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader();writer.writerows(manifest)
    with (directory/'unmeasured-sites.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        fields = list(pending_manifest[0]) if pending_manifest else ['image','site_index','nominal_line',
                 'pitch_index','predicted_x_px','predicted_y_px','native_accepted','reason']
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(pending_manifest)
    (directory/'annotation-coverage.json').write_text(json.dumps(coverage,indent=2))
    (directory/f'annotation-coverage{suffix}.json').write_text(json.dumps(coverage,indent=2))
    return dict(labels_per_image=visible_counts, threshold_px=floor,threshold_inclusive=inclusive,relation=relation, files=[
        f'01-bad-holes-grid-offsets{suffix}.png', f'02-good-holes-grid-offsets{suffix}.png'],
        manifest=f'annotated-pair-labels{suffix}.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--legacy',type=Path)
    parser.add_argument('--ignore-offset-at-most',type=float,default=1.0,
                        help='Label every measurable hole above this absolute offset independently per image')
    parser.add_argument('--min-offset',type=float,default=None,
                        help='Inclusive label threshold; overrides --ignore-offset-at-most')
    args=parser.parse_args();directory=args.directory
    if not np.isfinite(args.ignore_offset_at_most) or args.ignore_offset_at_most < 0:
        parser.error('--ignore-offset-at-most must be a finite nonnegative number')
    if args.min_offset is not None and (not np.isfinite(args.min_offset) or args.min_offset < 0):
        parser.error('--min-offset must be a finite nonnegative number')
    result=json.loads((directory/'results.json').read_text())
    rows=read_rows(directory/'per-hole-lines.csv')
    pair_annotation=render_annotated_pair(directory,result,rows,args.ignore_offset_at_most,args.min_offset)
    groups={line['line']:[r for r in rows if r['line']==line['line']] for line in result['lines']}
    image=cv2.imread(result['input_images'][0]);overlay=image.copy()
    # Draw only ranges actually supported by visible holes, not hidden bottom edges.
    for name,curve in result['curves'].items():
        p=np.polynomial.Polynomial(curve['bad_coeff']);u=np.linspace(*curve['range'],700);v=p(u)
        pts=np.column_stack([u,v] if name[0]=='H' else [v,u]).round().astype(np.int32)
        cv2.polylines(overlay,[pts],False,(220,130,0),2,cv2.LINE_AA)
        x,y=pts[0];cv2.putText(overlay,name,(int(x-28),int(y-20)),cv2.FONT_HERSHEY_SIMPLEX,.75,(150,60,0),2,cv2.LINE_AA)
    highlights=[]
    for row in rows:
        if row['bad_normal_to_curve_px'] is not None and abs(row['bad_normal_to_curve_px'])>=5 and row['reference_pair_reliable']:
            highlights.append(row)
            cx,cy=round(row['bad_x_px']),round(row['bad_y_px'])
            cv2.circle(overlay,(cx,cy),15,(35,40,223),3,cv2.LINE_AA)
    cv2.imwrite(str(directory/'curved-grid-overlay.png'),overlay)
    font_path='/System/Library/Fonts/STHeiti Medium.ttc'
    font=lambda size:ImageFont.truetype(font_path,size)
    poster=Image.new('RGB',(1100,1010),'#f0f4f5');draw=ImageDraw.Draw(poster)
    draw.text((28,20),'齒孔偏移｜以整條彎曲格線作基準',font=font(30),fill='#163445')
    draw.text((28,65),'藍線：完整曲線基準   紅圈：實測孔心   右側：整條線各孔的偏移（原圖像素）',font=font(19),fill='#47616d')
    examples=[('V6',11),('H5',72),('H3',57)]
    for n,(name,k) in enumerate(examples):
        row=next(r for r in groups[name] if r['pitch_index']==k)
        y0=110+n*290;value=row['bad_normal_to_curve_px'];direction='向右' if name[0]=='V' else '向上'
        draw.text((28,y0),f'{name} 第 {k} 孔距位置  |  {direction} {abs(value):.1f} px',font=font(23),fill='#163445')
        cx,cy=round(row['bad_x_px']),round(row['bad_y_px']);left,top=cx-50,cy-34
        crop=image[top:top+69,left:left+101].copy()
        curve=np.polynomial.Polynomial(result['curves'][name]['bad_coeff'])
        if name[0]=='H':
            u=np.arange(left,left+101);pts=np.column_stack([u-left,curve(u)-top])
        else:
            u=np.arange(top,top+69);pts=np.column_stack([curve(u)-left,u-top])
        # Pixel content is enlarged without interpolation; annotations do not alter source images.
        enlarged=cv2.resize(crop,(404,276),interpolation=cv2.INTER_NEAREST)
        cv2.polylines(enlarged,[np.round((pts+.5)*4-.5).astype(np.int32)],False,(220,130,0),2,cv2.LINE_AA)
        cv2.circle(enlarged,(202,138),28,(30,35,218),2,cv2.LINE_AA)
        poster.paste(Image.fromarray(enlarged[:,:,::-1]).resize((323,221)),(28,y0+40))
        gx,gy,gw,gh=410,y0+59,640,162
        members=groups[name];minimum=min(r['pitch_index'] for r in members);maximum=max(r['pitch_index'] for r in members)
        xp=lambda index:gx+(index-minimum)/(maximum-minimum)*gw
        yp=lambda delta:gy+(8-delta)/16*gh
        for val in [-5,0,5]:
            yy=yp(val);draw.line([(gx,yy),(gx+gw,yy)],fill='#c1ced3',width=1)
            draw.text((gx-34,yy-8),f'{val:+d}',font=font(14),fill='#647986')
        for rr in members:
            v=rr['bad_normal_to_curve_px']
            if v is None:continue
            xx,yy=xp(rr['pitch_index']),yp(v);c='#c8323b' if abs(v)>=5 else '#286b89'
            draw.ellipse((xx-2.5,yy-2.5,xx+2.5,yy+2.5),fill=c)
        for index in np.linspace(minimum,maximum,6):draw.text((xp(index)-10,gy+gh+8),f'{index:.0f}',font=font(14),fill='#647986')
        draw.text((gx,gy+gh+34),'孔距索引；正值：橫線向下／直線向右',font=font(16),fill='#647986')
    draw.text((28,984),'曲線補償不是鏡頭標定；底邊遮擋區不推算缺陷。',font=font(16),fill='#536c79')
    poster.save(directory/'curve-deviation-examples.png')
    benchmark=result['benchmark'];ms=benchmark['pair_median_ms'];compute=benchmark['in_memory_pair_median_ms']
    old=json.loads((args.legacy/'results.json').read_text())['benchmark']['pair_median_ms'] if args.legacy else None
    summary_rows=[];sections=[]
    for line in result['lines']:
        name=line['line'];row=groups[name]
        summary_rows.append(f'<tr><td><a href="#{name}">{name}</a></td><td>{line["accepted_bad"]}/{line["nominal_sites"]}</td><td>{line["median_reference_normal_px"]:+.2f}</td><td>{line["max_abs_hole_to_curve_px"]:.2f}</td><td>{line["reassigned_pitch_indices"]}</td></tr>')
        segments=' / '.join(f'{s["segment"]}: {s["median_normal_px"]:+.1f}' for s in line['segments'])
        sections.append(f'<section id="{name}"><h2>{name}　整條線位移 {line["median_reference_normal_px"]:+.2f} px</h2><p>在排除此線後，用其他格線對齊良品，再取此線對應孔的法向差中位數。藍點：壞圖；綠點：良圖；紅點：壞圖距自身完整曲線 ≥ 5 px（示範門檻）。</p>{plot_svg(row)}<p>每一枚郵票邊界區段的中位位移（px）：{segments}</p><p>本線 {line["paired"]} 個可靠對應；孔序重配 {line["reassigned_pitch_indices"]} 個。無影像支持的位置留空。</p></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>郵票完整格線與偏移量</title><style>body{max-width:1040px;margin:auto;padding:28px 18px;background:#f1f5f6;color:#183745;font:16px/1.6 -apple-system,BlinkMacSystemFont,sans-serif}h1{margin-bottom:8px}h2{font-size:21px}section{background:white;border:1px solid #d5e0e5;border-radius:8px;padding:18px;margin:18px 0}table{border-collapse:collapse;width:100%}th,td{text-align:right;border-bottom:1px solid #dce5e8;padding:6px 12px}th:first-child,td:first-child{text-align:left}a{color:#006791}svg,img{display:block;width:100%}section:target{border-color:#006791}p{margin:10px 0}</style><main><h1>郵票完整格線與偏移量</h1>'''
    page+=f'<p>兩張原圖，6 條橫線與 11 條直線。單孔量測：到完整彎曲線的最短距離；整線量測：對齊良品後的法向位移中位數。H 從上往下，V 從左往右；單位均為原圖像素。</p><section><h2>本次流程時間</h2><p>兩張讀圖＋偵測＋格線分析：<b>{ms/1000:.2f} 秒</b>；影像已在記憶體時，實測運算階段合計 <b>{compute/1000:.2f} 秒</b>。{benchmark["measured_repeats"]} 次紀錄的中位數，另排除一次暖機；不含取像、畫面顯示、輸出報告。</p>'
    if old:page+=f'<p>同版分析、舊輪廓流程：{old/1000:.2f} 秒（三次中位數）。本次加速 {old/ms:.1f} 倍；兩張圖的候選孔與原始模板量測陣列皆與舊試驗一致。</p>'
    page+='</section><section><h2>怎麼解讀畸變</h2><p>使用多數孔的緩慢曲線作基準，避免直接套一直線。對單條線做良品對齊時排除該線，避免把整線偏移吸收到基準。這只是影像上的形變補償，不能由這兩張圖分離鏡頭畸變、紙張翹曲與緩慢的大範圍缺陷；毫米需另有標定。</p><p>底邊部分孔被黑色邊框遮住，無法完整量測。右邊 V11 的末段有 10 個孔序重新對應；先前 #13、#14 的「沿線偏移」不宜直接沿用。</p></section>'
    page+='<section><h2>17 條線摘要</h2><table><tr><th>格線</th><th>可量／名義孔位</th><th>整線位移 px</th><th>最大單孔法向偏移 px</th><th>孔序重配</th></tr>'+''.join(summary_rows)+'</table></section>'
    page+='<p><a href="curved-grid-overlay.png">原解析度曲線標記圖</a> · <a href="per-hole-lines.csv">每個孔的完整數值 CSV</a> · <a href="results.json">模型與計時原始紀錄</a></p>'
    page+=f'<section><h2>所有達門檻孔位，不限制先前 27 點</h2><p>各圖獨立標出所有 {pair_annotation["relation"]} {pair_annotation["threshold_px"]:g} px 的已量測孔位；圖一 {pair_annotation["labels_per_image"][0]} 個、圖二 {pair_annotation["labels_per_image"][1]} 個。這是模型殘差，不是已確認缺陷數。交點在圖上標較大的分量，CSV 保留每條線的量測。</p><p><a href="{pair_annotation["files"][0]}">圖一：全部齒孔偏移數值</a> · <a href="{pair_annotation["files"][1]}">圖二：全部齒孔偏移數值</a> · <a href="{pair_annotation["manifest"]}">全部超門檻量測 CSV</a> · <a href="unmeasured-sites.csv">未可靠量測名義孔位</a> · <a href="annotation-coverage.json">標註完整性紀錄</a></p></section>'
    page+=''.join(sections)+'</main></html>'
    (directory/'review-lines.html').write_text(page)
    with (directory/'line-summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        fields=[k for k in result['lines'][0] if k!='segments'];writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows({k:l[k] for k in fields} for l in result['lines'])
    print(json.dumps(dict(lines=len(groups),shown_perpendicular_candidates=len(highlights),pair_ms=ms,in_memory_ms=compute,pair_annotation=pair_annotation)))


if __name__=='__main__':
    main()
