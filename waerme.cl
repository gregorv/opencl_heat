
int get_idx(int x, int y, int2 res)
{
	return x*res.y + y;
}

__kernel void calc_gradient(__global float* skalarField, __global float2* gradientField, int2 res)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int xp = x >= res.x-1? x : x+1;
	int xm = x <= 0?     x : x-1;
	int yp = y >= res.y? y : y+1;
	int ym = y <= 0?     y : y-1;
	float dxp = skalarField[xp*res.y + y];
	float dxm = skalarField[xm*res.y + y];
	float dyp = skalarField[x*res.y  + yp];
	float dym = skalarField[x*res.y  + ym];
	gradientField[x*res.y + y] = (float2)((dxp-dxm)/2.0f, (dyp-dym)/2.0f);
}

__kernel void solve_heat_equation(__global float* iTemperature, __global float* oTemperature, __global float2* gradientField, int2 resolution, __write_only image2d_t dest)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int xp = x>=resolution.x-1? x : x+1;
	int xm = x<=0? x : x-1;
	int yp = y>=resolution.y? y : y+1;
	int ym = y<=0? y : y-1;
	float T = iTemperature[x * resolution.y + y];
	float Txp = iTemperature[xp * resolution.y + y];
	float Typ = iTemperature[x * resolution.y + yp];
	float Txm = iTemperature[xm * resolution.y + y];
	float Tym = iTemperature[x * resolution.y + ym];
	float d = 1e-2f;
	float div = (Txp - 2.f*T + Txm + Typ - 2.f*T + Tym)/(d*d);
	oTemperature[x*resolution.y + y] = iTemperature[x*resolution.y + y] + div*0.00001f;
	if(oTemperature[x*resolution.y + y] < 0.0f)
		oTemperature[x*resolution.y + y] = 0.0f;
	write_imagef(dest, (int2)(x,y), (float4)(oTemperature[x*resolution.y + y]*1e-3f, -oTemperature[x*resolution.y + y]*1e-3f, 0.0f, 1.0f));
	//write_imagef(dest, (int2)(x,y), (float4)(gradientField[x*resolution.y + y].x, gradientField[x*resolution.y + y].y, 0.0f, 1.0f));
	//write_imagef(dest, (int2)(x,y), (float4)(div, -div, 0.0f, 1.0f));
	//write_imagef(dest, (int2)(x,y), (float4)(1.0f*x/resolution.x, 1.0f*y/resolution.y, 0.0f, 1.0f));
}