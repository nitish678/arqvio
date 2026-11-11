// The ARQVIO mark: six blades, red and silver alternating, as on the website -- drawn with token fills (no
// gradient, no raster), so it stays sharp at any size.

const BLADE = "M15.8 -2.8L20.4 -6.3L23.3 -10.5L25.2 -15.3L26.1 -20.6L25.8 -26.2L24.5 -32.0L22.1 -37.7L18.6 -43.2L14.1 -48.3L8.5 -52.9L2.1 -56.7L-5.2 -59.6L-13.2 -61.5L-21.7 -62.3L-30.5 -61.9L-39.5 -60.1L-48.6 -57.0L-57.4 -52.6L-65.8 -46.8L-73.6 -39.7L-80.6 -31.4L-86.7 -21.9L-91.5 -11.4L-95.0 0L-87.1 -14.5L-77.7 -26.1L-67.3 -35.2L-56.4 -41.7L-45.3 -45.7L-34.7 -47.6L-24.8 -47.4L-15.9 -45.7L-8.3 -42.7L-2.0 -38.9L3.0 -34.5L6.6 -29.8L9.2 -25.3L10.7 -20.9L11.5 -17.0L11.8 -13.6L11.6 -10.8L11.3 -8.4L11.0 -6.6L10.8 -5.2L10.9 -4.1L11.4 -3.3L12.6 -2.7Z";

export function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="-100 -100 200 200" aria-hidden="true" focusable="false">
      <g strokeWidth="1.6" strokeLinejoin="round">
        {[0, 60, 120, 180, 240, 300].map((deg, i) => (
          <path key={deg} d={BLADE} className={i % 2 === 0 ? "blade-red" : "blade-silver"} transform={`rotate(${deg})`} />
        ))}
      </g>
    </svg>
  );
}
