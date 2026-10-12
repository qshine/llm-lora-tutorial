"""生成片尾二维码（指向在线电子书）。"""
import qrcode
q = qrcode.QRCode(border=0, error_correction=qrcode.constants.ERROR_CORRECT_M)
q.add_data('https://qshine.github.io/llm-lora-tutorial/'); q.make()
m = q.get_matrix(); n = len(m)
d = ''.join(f'M{x} {y}h1v1h-1z' for y, r in enumerate(m) for x, v in enumerate(r) if v)
open('qr.svg', 'w').write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" shape-rendering="crispEdges"><path d="{d}" fill="#141614"/></svg>')
