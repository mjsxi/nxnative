// The present: the frame drawn into the drawable, cropped (nvn_mtl.m present_pipeline)
#include <metal_stdlib>
using namespace metal;
struct V { float4 p [[position]]; float2 uv; };
vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uv; return o; }
struct Crop { float4 rect; float4 edge; };
fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant Crop& c [[buffer(0)]]) { return t.sample(s, clamp(in.uv * c.rect.zw + c.rect.xy, c.edge.xy, c.edge.zw)); }

// Antialiasing for the same draw: DM_PRESENT_AA=fxaa makes present_pipeline pick this over fs above (nvn_mtl.m).
// Lottes' console-quality FXAA core. The source is the swapchain's raw bytes (present_raw_src takes a non-sRGB view of an sRGB
// texture, because the Switch's scan-out sends the stored bytes as they are), so the luma below is gamma-encoded, which is
// where this filter belongs. Everything outside a luminance edge takes the early out and returns the source texel unchanged,
// so flat areas, gradients and the interior of text come out as they were.
static float fx_lum(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }
fragment float4 fs_fxaa(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant Crop& c [[buffer(0)]]) {
    constexpr sampler q(coord::normalized, address::clamp_to_edge, filter::nearest);
    float2 uv = clamp(in.uv * c.rect.zw + c.rect.xy, c.edge.xy, c.edge.zw);
    float2 px = 1.0 / float2(t.get_width(), t.get_height());
    float3 rgbM = t.sample(q, uv).rgb;
    float lM = fx_lum(rgbM);
    float lN = fx_lum(t.sample(q, uv + float2(0.0, -px.y)).rgb);
    float lS = fx_lum(t.sample(q, uv + float2(0.0, px.y)).rgb);
    float lE = fx_lum(t.sample(q, uv + float2(px.x, 0.0)).rgb);
    float lW = fx_lum(t.sample(q, uv - float2(px.x, 0.0)).rgb);
    float lMin = min(lM, min(min(lN, lS), min(lE, lW)));
    float lMax = max(lM, max(max(lN, lS), max(lE, lW)));
    if (lMax - lMin < max(1.0 / 32.0, lMax * 0.125)) return float4(rgbM, 1.0);   // not an edge
    float lNW = fx_lum(t.sample(q, uv + float2(-px.x, -px.y)).rgb);
    float lNE = fx_lum(t.sample(q, uv + float2(px.x, -px.y)).rgb);
    float lSW = fx_lum(t.sample(q, uv + float2(-px.x, px.y)).rgb);
    float lSE = fx_lum(t.sample(q, uv + float2(px.x, px.y)).rgb);
    float2 dir = float2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
    float dirReduce = max((lNW + lNE + lSW + lSE) * 0.03125, 1.0 / 128.0);   // 1/4 of the four, times the reduce multiply
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(float2(8.0), max(float2(-8.0), dir * rcpDirMin)) * px;   // at most 8 texels along the edge
    float2 uvA = uv + dir * (1.0 / 3.0 - 0.5), uvB = uv - dir * (2.0 / 3.0 - 0.5);
    float3 rgbA = 0.5 * (t.sample(q, uvA).rgb + t.sample(q, uvB).rgb);
    float3 rgbB = rgbA * 0.5 + 0.25 * (t.sample(q, uvA - dir * 0.5).rgb + t.sample(q, uvB + dir * 0.5).rgb);
    float lB = fx_lum(rgbB);
    return float4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);   // keep the blend inside the local range
}
