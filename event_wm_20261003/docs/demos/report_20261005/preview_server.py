"""Local-only preview server with byte ranges for video seeking in WebKit."""
import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        path=self.translate_path(self.path)
        if os.path.isdir(path) or not self.headers.get('Range'):
            return super().send_head()
        try:
            f=open(path,'rb'); size=os.fstat(f.fileno()).st_size
            raw=self.headers['Range']
            if not raw.startswith('bytes=') or ',' in raw: raise ValueError('Unsupported range')
            lo,hi=raw[6:].split('-',1)
            start=int(lo) if lo else max(0,size-int(hi))
            end=min(int(hi) if hi else size-1,size-1)
            if start>end or start>=size: raise ValueError('Invalid range')
            self.send_response(206); self.send_header('Content-type',self.guess_type(path))
            self.send_header('Accept-Ranges','bytes'); self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.send_header('Content-Length',str(end-start+1)); self.end_headers()
            f.seek(start); self.range_remaining=end-start+1; return f
        except FileNotFoundError:
            self.send_error(404); return None
        except ValueError:
            self.send_error(416); return None

    def copyfile(self,source,outputfile):
        remaining=getattr(self,'range_remaining',None)
        if remaining is None: return super().copyfile(source,outputfile)
        while remaining:
            chunk=source.read(min(65536,remaining))
            if not chunk: break
            outputfile.write(chunk); remaining-=len(chunk)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--port',type=int,default=8770)
    args=ap.parse_args(); ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
