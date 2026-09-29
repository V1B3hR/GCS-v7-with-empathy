import React from 'react';

/**
 * High-performance SVG Sparkline with smooth Bézier curvature and gradient glow
 */
export const Sparkline = ({ data = [], color = '#10B981', height = 28, strokeWidth = 2 }) => {
  if (!data || data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min === 0 ? 1 : max - min;
  const width = 100; // SVG viewBox coordinates 0 to 100
  const padding = 3;

  const points = data.map((val, idx) => {
    const x = (idx / (data.length - 1)) * width;
    const y = height - padding - ((val - min) / range) * (height - padding * 2);
    return { x, y };
  });

  // Build smooth cubic Bézier curve path
  let pathD = `M ${points[0].x},${points[0].y}`;
  for (let i = 0; i < points.length - 1; i++) {
    const current = points[i];
    const next = points[i + 1];
    const controlX = (current.x + next.x) / 2;
    pathD += ` C ${controlX},${current.y} ${controlX},${next.y} ${next.x},${next.y}`;
  }

  // Build area fill path
  const areaD = `${pathD} L ${width},${height} L 0,${height} Z`;
  const gradientId = `spark-grad-${color.replace('#', '')}`;

  return (
    <div className="sparkline-container">
      <svg
        className="sparkline-svg"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.35" />
            <stop offset="100%" stopColor={color} stopOpacity="0.0" />
          </linearGradient>
        </defs>
        <path d={areaD} fill={`url(#${gradientId})`} />
        <path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Pulsing latest point */}
        <circle
          cx={points[points.length - 1].x}
          cy={points[points.length - 1].y}
          r={2.5}
          fill="#ffffff"
          stroke={color}
          strokeWidth={1.5}
        />
      </svg>
    </div>
  );
};
