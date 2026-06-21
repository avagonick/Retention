import { useRef, useEffect } from "react";

// Looping background video, sampled into a canvas at low resolution then
// upscaled with smoothing OFF -> a true pixelated effect. The <video> is kept
// full-size and playing BEHIND the canvas (covered, not display:none — a hidden
// video gets frozen by the browser), and we call play() explicitly so it loops.
export default function BackgroundVideo({ src = "/assets/background.mp4", pixelSize = 8 }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    const small = document.createElement("canvas");
    const sctx = small.getContext("2d");

    video.muted = true;
    const tryPlay = () => {
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
    };
    tryPlay();
    video.addEventListener("loadeddata", tryPlay);
    video.addEventListener("canplay", tryPlay);

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    let raf;
    const draw = () => {
      if (video.readyState >= 2 && canvas.width > 0) {
        const w = Math.max(1, Math.floor(canvas.width / pixelSize));
        const h = Math.max(1, Math.floor(canvas.height / pixelSize));
        small.width = w;
        small.height = h;
        sctx.drawImage(video, 0, 0, w, h); // downscale current frame
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(small, 0, 0, w, h, 0, 0, canvas.width, canvas.height); // upscale, blocky
      }
      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      video.removeEventListener("loadeddata", tryPlay);
      video.removeEventListener("canplay", tryPlay);
    };
  }, [pixelSize]);

  return (
    <div className="bg-layer" aria-hidden="true">
      <video ref={videoRef} className="bg-source" src={src} loop muted playsInline preload="auto" autoPlay />
      <canvas ref={canvasRef} className="bg-video" />
    </div>
  );
}
