// Usage: Drag with the mouse to add smoke to the fluid. This will also move a "rotor" that disturbs
//        the velocity field at the mouse location. Press the indicated keys to change options
//--------------------------------------------------------------------------------------------------

// #include <rfftw.h>              //the numerical simulation FFTW library
#include <fftw/fftw3.h>
// #include <pocketfft/pocketfft.h>
#include <stdio.h>              //for printing the help text
#include <math.h>               //for various math functions
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <immintrin.h>

using namespace std;


//--- SIMULATION PARAMETERS ------------------------------------------------------------------------
const int DIM = 52;				//size of simulation grid
const double dt = 0.2;				//simulation time step
float visc = 0.001;				//fluid viscosity
double *vx, *vy;             //(vx,vy)   = velocity field at the current moment
double *vx0, *vy0;           //(vx0,vy0) = velocity field at the previous moment
double *fx, *fy;	            //(fx,fy)   = user-controlled simulation forces, steered with the mouse
double *rho, *rho0;			//smoke density at the current (rho) and previous (rho0) moment
// rfftwnd_plan plan_rc, plan_cr;  //simulation domain discretization

fftw_plan plan_r2c_vx0;
fftw_plan plan_c2r_vx0;   

fftw_plan plan_r2c_vy0;
fftw_plan plan_c2r_vy0;   


//--- VISUALIZATION PARAMETERS ---------------------------------------------------------------------
int   winWidth, winHeight;      //size of the graphics window, in pixels
int   color_dir = 0;            //use direction color-coding or not
float vec_scale = 52;			//scaling of hedgehogs
int   draw_smoke = 1;           //draw the smoke or not
int   draw_vecs = 0;            //draw the vector field or not
const int COLOR_BLACKWHITE=0;   //different types of color mapping: black-and-white, rainbow, banded
const int COLOR_RAINBOW=1;
const int COLOR_BANDS=2;
int   scalar_col = COLOR_BLACKWHITE;           //method for scalar coloring
int   frozen = 0;               //toggles on/off the animation


int clamp(float x)
{ return ( (x) >= 0.0? ((int)(x)) : (-((int)(1-(x) ) ) ) ); }


//------ SIMULATION CODE STARTS HERE -----------------------------------------------------------------

//init_simulation: Initialize simulation data structures as a function of the grid size 'n'.
//                 Although the simulation takes place on a 2D grid, we allocate all data structures as 1D arrays,
//                 for compatibility with the FFTW numerical library.
void init_simulation(int n)
{
	int i; size_t dim;
	std::cout << "n: " << n << '\n';
	assert(n % 4 == 0);

	double clamp_test = 100.0f;
	int result = clamp(clamp_test);
	double clamp_neg_test = -100.0f;
	int neg_result = clamp(clamp_neg_test);
	std::cout << "clamp pos: " << result << ", clamp neg: " << neg_result << '\n';

	dim     = n * 2*(n/2+1)*sizeof(double);        //Allocate data structures
	std::cout << "number of data points: " << dim << '\n';
	vx       = (double*) malloc(dim);
	vy       = (double*) malloc(dim);
	vx0      = (double*) malloc(dim);
	vy0      = (double*) malloc(dim);
	dim     = n * n * sizeof(double);
	fx      = (double*) malloc(dim);
	fy      = (double*) malloc(dim);
	rho     = (double*) malloc(dim);
	rho0    = (double*) malloc(dim);
	

	for (i = 0; i < n * n; i++)                      //Initialize data structures to 0
	{
		vx[i] = vy[i] = vx0[i] = vy0[i] = fx[i] = fy[i] = rho[i] = rho0[i] = 0.0;
	}


	plan_r2c_vx0 = fftw_plan_dft_r2c_2d(DIM, DIM, vx0, static_cast<fftw_complex *>(static_cast<void *>(vx0)), FFTW_DESTROY_INPUT);
    plan_c2r_vx0 = fftw_plan_dft_c2r_2d(DIM, DIM, static_cast<fftw_complex *>(static_cast<void *>(vx0)), vx0, FFTW_DESTROY_INPUT);

    plan_r2c_vy0 = fftw_plan_dft_r2c_2d(DIM, DIM, vy0, static_cast<fftw_complex *>(static_cast<void *>(vy0)), FFTW_DESTROY_INPUT);
    plan_c2r_vy0 = fftw_plan_dft_c2r_2d(DIM, DIM, static_cast<fftw_complex *>(static_cast<void *>(vy0)), vy0, FFTW_DESTROY_INPUT);


}



float max(float x, float y)
{ return x > y ? x : y; }

//solve: Solve (compute) one step of the fluid flow simulation
void solve(int n, double* vx, double* vy, double* vx0, double* vy0, double visc, double dt)
{
	double  x0, y0, f, r, U[2], V[2], s, t;
	int  i0, j0, i1, j1;

	double vec_dt[4] = {dt,dt,dt,dt};
	double vec_zeros[4] = {0.0,0.0,0.0,0.0};	
	const size_t buffer_byte_count = n * 2*(n/2+1)*sizeof(double);   
	constexpr size_t simd_byte_count = 4 * sizeof(double);
	
	// update v0, initialize v.
	// V += dt * V0;
	for (int idx = 0; idx < n*n; idx += 4)
	{
		// vx[idx] += dt*vx0[idx];
		// vx0[idx] = vx[idx];		
		// vy[idx] += dt*vy0[idx];
		// vy0[idx] = vy[idx];



		// we DON'T! actually require intermediate?
		// __m256d vx_idx = _mm256_fmadd_pd( *(__m256d*)&vx0[idx], *(__m256d*)vec_dt, *(__m256d*)&vx[idx]);
		// __m256d vy_idx = _mm256_fmadd_pd( *(__m256d*)&vy0[idx], *(__m256d*)vec_dt, *(__m256d*)&vy[idx]);
		// _mm256_storeu_pd (&vx[idx], vx_idx);
		// _mm256_storeu_pd (&vy[idx], vy_idx);
		// _mm256_storeu_pd (&vx0[idx], *(__m256d*)&vx[idx]);
		// _mm256_storeu_pd (&vy0[idx], *(__m256d*)&vy[idx]);

		// vx = vx0 * dt + vx;
		*(__m256d*)&vx[idx] = _mm256_fmadd_pd( *(__m256d*)&vx0[idx], *(__m256d*)vec_dt, *(__m256d*)&vx[idx]);
		*(__m256d*)&vy[idx] = _mm256_fmadd_pd( *(__m256d*)&vy0[idx], *(__m256d*)vec_dt, *(__m256d*)&vy[idx]);
	}

	// V0 = V;
	memcpy((void*)(&vx0[0]),(void*)(&vx[0]), n * n * sizeof(double));
	memcpy((void*)(&vy0[0]),(void*)(&vy[0]), n * n * sizeof(double));


	constexpr double start_value = 0.5f / DIM;
	double x = start_value;
	double y = start_value;
	double vec_x[4] = {start_value, start_value,start_value, start_value};	
	double vec_y[4] = {start_value, start_value,start_value, start_value};


	constexpr double increment = 1.0f / DIM;
	const double vec_neg_dt[4] = {-dt, -dt, -dt, -dt};
	const double vec_n[4] = {DIM, DIM, DIM, DIM};
	const double vec_minus_half[4] = {-0.5f,-0.5f,-0.5f,-0.5f};
	const double vec_increment[4] = {increment, increment, increment, increment};
	const uint64_t vec_ui_zeroes[4] = {0,0,0,0};
	const uint64_t vec_ui_ones[4] = {1,1,1,1};


	for (int i = 0; i < n ; ++i)
	{
		for (int j = 0;  j < n; j += 4)
		{	
			int cell_idx = i + n * j;
			x0 = n * (x - dt * vx0[cell_idx]) -0.5f;
			y0 = n * (y - dt * vy0[cell_idx]) -0.5f;
			// {

				// intermeditate = -dt * v + a)
				__m256d intermediate =  _mm256_fmadd_pd( *(__m256d*)&vx0[cell_idx], *(__m256d*)vec_neg_dt, *(__m256d*)&vec_x);
				// // n * intermediate - 0.5f)
				__m256d vec_x0 = _mm256_fmadd_pd(intermediate, *(__m256d*)vec_n, *(__m256d*)&vec_minus_half);

				// // intermeditate = -dt * v + a)
				intermediate = _mm256_fmadd_pd( *(__m256d*)&vy0[cell_idx], *(__m256d*)vec_neg_dt, *(__m256d*)&vec_y);
				// // n * intermediate - 0.5f)
				__m256d vec_y0 = _mm256_fmadd_pd(intermediate, *(__m256d*)vec_n, *(__m256d*)&vec_minus_half);


			// } 
			      	
			// convert to int and clamp.
	      	i0 = clamp(x0); // double to int?
	      	s = x0 - i0;
    	  	i0 = (n + (i0 % n)) % n;
	      	i1 = (i0 + 1) % n;

			
			// {
			// clamp
				__m256i vec_i0 =  _mm256_cvtpd_epu64(vec_x0);
				__mmask8 negative_mask = _mm256_cmplt_epu64_mask(vec_x0,  *(__m256d*)vec_ui_zeroes); // if numbers are negative, make it true.
				vec_i0 = _mm256_mask_sub_epi64(vec_i0, negative_mask, vec_i0,*(__m256d*)vec_ui_ones);

				__m256i vec_s = __m256_sub_epi64(vec_x0, vec_i0);
				vec_i0 =  _mm256_rem_epi64 (vec_i0, vec_n);
				vec_i0 =  _mm256_add_epi64(vec_i0, vec_n);
				vec_i0 =  _mm256_rem_epi64 (vec_i0, vec_n);
				
				__mm256 vec_i1 =_mm256_add_epi64(vec_i0, vec_ui_ones);
				vec_i1 = _mm256_rem_epi64(veC_i1, vec_n);
			// }



	      	j0 = clamp(y0);
	      	t = y0 - j0;
	      	j0 = (n +(j0 % n)) % n;
	      	j1 = (j0 + 1) % n;



			// clamp
			// {
	      		// clamp
				__m256i vec_j0 =  _mm256_cvtpd_epu64(vec_y0);
				__mmask8 negative_mask = _mm256_cmplt_epu64_mask(
					vec_y0,
					*(__m256d*)vec_ui_zeroes); // if numbers are negative, make it true.
				vec_j0 = _mm256_mask_sub_epi64(vec_j0, negative_mask, vec_j0,*(__m256d*)vec_ui_ones);
				
				__m256i vec_t = __m256_sub_epi64(vec_y0, vec_j0);
				

		     	vec_j0 =  _mm256_rem_epi64 (vec_j0, vec_n);
				vec_j0 =  _mm256_add_epi64(vec_j0, vec_n);
				vec_j0 =  _mm256_rem_epi64 (vec_j0, vec_n);
				
				__mm256i vec_j1 =_mm256_add_epi64(vec_j0, vec_ui_ones);
				vec_j1 = _mm256_rem_epi64(vec_j1, vec_n);

			// }

	      	int lower_clamp_idx = i0 + n * j0;
	      	int col_clamp_idx = i0 + n * j1;
	      	int row_clamp_idx = i1 + n * j0;
	      	int upper_clamp_idx = i1 + n * j1;

	      	// {
	      	__m256i vec_lower_clamp_idx = _mm256_mullo_epi64(vec_n, vec_j0);
	      	__m256i vec_col_clamp_idx =  _mm256_mullo_epi64(vec_n, vec_j1);		
	      	__m256i vec_row_clamp_idx =   _mm256_mullo_epi64(vec_n,vec_j0);		
	      	__m256i vec_upper_clamp_idx  = _mm256_mullo_epi64(vec_n,vec_j1);	
		
			vec_lower_clamp_idx = _mm256_add_epi64(vec_i0, vec_lower_clamp_idx);
			vec_col_clamp_idx =  _mm256_add_epi64(vec_i0, vec_col_clamp_idx);	
			vec_row_clamp_idx =   _mm256_add_epi64(vec_i1,vec_row_clamp_idx);	
			vec_upper_clamp_idx  = _mm256_add_epi64(vec_i1,vec_upper_clamp_idx);



	       // }

	      	vx[cell_idx] = (1 - s) * ((1 - t) * vx0[lower_clamp_idx] +
	      				    t * vx0[col_clamp_idx]) + 
	      					s * ((1 - t) * vx0[row_clamp_idx] + 
      						t * vx0[upper_clamp_idx]);

	      	vy[cell_idx] = (1 - s) * ((1 - t) * vy0[lower_clamp_idx] + 
	      					t * vy0[col_clamp_idx]) + 
	      					s * ((1 - t) * vy0[row_clamp_idx] + 
      						t * vy0[upper_clamp_idx]);
		     
		    y += increment; 
		   }

	   x += increment;
	}
	  
	int accumulator = 0;
	// ?? what term is this?
	for(int row_idx = 0; row_idx < n; ++row_idx)
	{
		for(int col_idx =0; col_idx < n; ++col_idx)
	 	{ 
	 		//
 			vx0[row_idx + (n + 2) * col_idx] = vx[row_idx  + n * col_idx]; 
 			vy0[row_idx + (n + 2) * col_idx] = vy[row_idx  + n * col_idx];
 		}
	}

	// assert(accumulator == 49 * 49);
	  
    fftw_execute(plan_r2c_vx0);
    fftw_execute(plan_r2c_vy0);

    // why are we skipping steps here?
	for (int row_idx =0; row_idx <= n; row_idx += 2)
	{
	   x = 0.5f * row_idx;
	   for (int col_idx = 0; col_idx < n ; col_idx++)
	   {
	      y = col_idx <= n/2 ? (double)col_idx : (double)col_idx - n;
	      r =  x * x + y * y;
	      if ( r == 0.0f ) continue;

	      f = (double)exp(-r * dt * visc);


	      U[0] = vx0[row_idx  +(n+2)*col_idx]; V[0] = vy0[row_idx  +(n+2)*col_idx];
	      U[1] = vx0[row_idx+1+(n+2)*col_idx]; V[1] = vy0[row_idx+1+(n+2)*col_idx];

	      vx0[row_idx  +(n+2)*col_idx] = f*((1-x*x/r)*U[0]     -x*y/r *V[0]);
	      vx0[row_idx+1+(n+2)*col_idx] = f*((1-x*x/r)*U[1]     -x*y/r *V[1]);
	      vy0[row_idx+  (n+2)*col_idx] = f*(  -y*x/r *U[0] + (1-y*y/r)*V[0]);
	      vy0[row_idx+1+(n+2)*col_idx] = f*(  -y*x/r *U[1] + (1-y*y/r)*V[1]);
	   }
	}

    fftw_execute(plan_c2r_vx0);
    fftw_execute(plan_c2r_vy0);

	f = 1.0/(n*n);
 	for (int row_idx = 0; row_idx < n; row_idx++)
 	{
		for (int col_idx = 0; col_idx < n; col_idx++)
		{
			vx[row_idx + n * col_idx] = f*vx0[row_idx + ( n + 2) * col_idx]; vy[row_idx + n * col_idx] = f*vy0[row_idx + ( n +2)*col_idx]; 	
		} 		
 	}
}


// diffuse_matter: This function diffuses matter that has been placed in the velocity field. It's almost identical to the
// velocity diffusion step in the function above. The input matter densities are in rho0 and the result is written into rho.
void diffuse_matter(int n, double *vx, double *vy, double *rho, double *rho0, double dt)
{
	double x, y, x0, y0, s, t;
	int i, j, i0, j0, i1, j1;

	for ( x=0.5f/n,i=0 ; i<n ; i++,x+=1.0f/n )
		for ( y=0.5f/n,j=0 ; j<n ; j++,y+=1.0f/n )
		{
			x0 = n*(x-dt*vx[i+n*j])-0.5f;
			y0 = n*(y-dt*vy[i+n*j])-0.5f;
			i0 = clamp(x0);
			s = x0-i0;
			i0 = (n+(i0%n))%n;
			i1 = (i0+1)%n;
			j0 = clamp(y0);
			t = y0-j0;
			j0 = (n+(j0%n))%n;
			j1 = (j0+1)%n;
			rho[i+n*j] = (1-s)*((1-t)*rho0[i0+n*j0]+t*rho0[i0+n*j1])+s*((1-t)*rho0[i1+n*j0]+t*rho0[i1+n*j1]);
		}
}

//set_forces: copy user-controlled forces to the force vectors that are sent to the solver.
//            Also dampen forces and matter density to get a stable simulation.
void set_forces(void)
{
	int i;
	for (i = 0; i < DIM * DIM; i++)
	{
        rho0[i]  = 0.995 * rho[i];
        fx[i] *= 0.85;
        fy[i] *= 0.85;
        vx0[i]    = fx[i];
        vy0[i]    = fy[i];
	}
}


//do_one_simulation_step: Do one complete cycle of the simulation:
//      - set_forces:
//      - solve:            read forces from the user
//      - diffuse_matter:   compute a new set of velocities
//      - gluPostRedisplay: draw a new visualization frame
void do_one_simulation_step(void)
{
	if (!frozen)
	{
	  set_forces();
	  solve(DIM, vx, vy, vx0, vy0, visc, dt);
	  diffuse_matter(DIM, vx, vy, rho, rho0, dt);
	  // glutPostRedisplay();
	}
}


//------ VISUALIZATION CODE STARTS HERE -----------------------------------------------------------------


//rainbow: Implements a color palette, mapping the scalar 'value' to a rainbow color RGB
void rainbow(float value,float* R,float* G,float* B)
{
   const float dx=0.8;
   if (value<0) value=0; if (value>1) value=1;
   value = (6-2*dx)*value+dx;
   *R = max(0.0,(3-fabs(value-4)-fabs(value-5))/2);
   *G = max(0.0,(4-fabs(value-2)-fabs(value-4))/2);
   *B = max(0.0,(3-fabs(value-1)-fabs(value-2))/2);
}

//set_colormap: Sets three different types of colormaps
void set_colormap(float vy)
{
   float R,G,B;

   if (scalar_col==COLOR_BLACKWHITE)
       R = G = B = vy;
   else if (scalar_col==COLOR_RAINBOW)
       rainbow(vy,&R,&G,&B);
   else if (scalar_col==COLOR_BANDS)
       {
          const int NLEVELS = 7;
          vy *= NLEVELS; vy = (int)(vy); vy/= NLEVELS;
	      rainbow(vy,&R,&G,&B);
	   }

   glColor3f(R,G,B);
}


//direction_to_color: Set the current color by mapping a direction vector (x,y), using
//                    the color mapping method 'method'. If method==1, map the vector direction
//                    using a rainbow colormap. If method==0, simply use the white color
void direction_to_color(float x, float y, int method)
{
	float r,g,b,f;
	if (method)
	{
	  f = atan2(y,x) / 3.1415927 + 1;
	  r = f;
	  if(r > 1) r = 2 - r;
	  g = f + .66667;
      if(g > 2) g -= 2;
	  if(g > 1) g = 2 - g;
	  b = f + 2 * .66667;
	  if(b > 2) b -= 2;
	  if(b > 1) b = 2 - b;
	}
	else
	{ r = g = b = 1; }
	glColor3f(r,g,b);
}

//visualize: This is the main visualization function
void visualize(void)
{
	int        i, j, idx; double px,py;

	double  wn = (double)winWidth / (double)(DIM + 1);   // Grid cell width
	double  hn = (double)winHeight / (double)(DIM + 1);  // Grid cell heigh


	if (draw_smoke)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		for (j = 0; j < DIM - 1; j++)			//draw smoke
		{

			glBegin(GL_TRIANGLE_STRIP);

			i = 0;
			px = wn + (double)i * wn;
			py = hn + (double)j * hn;
			idx = (j * DIM) + i;

			glVertex2f(px,py);

			for (i = 0; i < DIM - 1; i++)
			{
				px = wn + (double)i * wn;
				py = hn + (double)(j + 1) * hn;
				idx = ((j + 1) * DIM) + i;
				set_colormap(rho[idx]);
				glVertex2f(px, py);
				px = wn + (double)(i + 1) * wn;
				py = hn + (double)j * hn;
				idx = (j * DIM) + (i + 1);
				set_colormap(rho[idx]);
				glVertex2f(px, py);
			}

			px = wn + (double)(DIM - 1) * wn;
			py = hn + (double)(j + 1) * hn;
			idx = ((j + 1) * DIM) + (DIM - 1);
			set_colormap(rho[idx]);
			glVertex2f(px, py);
			glEnd();
		}
	}

	if (draw_vecs)
	{
	  glBegin(GL_LINES);				//draw velocities
	  for (i = 0; i < DIM; i++)
	    for (j = 0; j < DIM; j++)
	    {
		  idx = (j * DIM) + i;
		  direction_to_color(vx[idx],vy[idx],color_dir);
		  glVertex2f(wn + (double)i * wn, hn + (double)j * hn);
		  glVertex2f((wn + (double)i * wn) + vec_scale * vx[idx], (hn + (double)j * hn) + vec_scale * vy[idx]);
	    }
	  glEnd();
	}
}


//------ INTERACTION CODE STARTS HERE -----------------------------------------------------------------

//display: Handle window redrawing events. Simply delegates to visualize().
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	visualize();
	glFlush();
}

//reshape: Handle window resizing (reshaping) events
void glfw_window_size_callback(GLFWwindow* window, int w, int h)
{
 	glViewport(0.0f, 0.0f, (GLfloat)w, (GLfloat)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, (GLdouble)w, 0.0, (GLdouble)h, -1.0, 1.0);

	winWidth = w; winHeight = h;
}



//@IC(Sjors): parameters cannot be const since the callbacks needs to match.
void glfw_cursor_position_callback(GLFWwindow* window, double mx, double my)
{
    int xi,yi,X,Y; double  dx, dy, len;
	static int lmx = 0, lmy = 0;				//remembers last mouse location

	// Compute the array index that corresponds to the cursor location
	xi = (int)clamp((double)(DIM + 1) * ((double)mx / (double)winWidth));
	yi = (int)clamp((double)(DIM + 1) * ((double)(winHeight - my) / (double)winHeight));

	X = xi; Y = yi;

	if (X > (DIM - 1))  X = DIM - 1; if (Y > (DIM - 1))  Y = DIM - 1;
	if (X < 0) X = 0; if (Y < 0) Y = 0;

	// Add force at the cursor location
	my = winHeight - my;
	dx = mx - lmx; dy = my - lmy;
	len = sqrt(dx * dx + dy * dy);
	if (len != 0.0) {  dx *= 0.1 / len; dy *= 0.1 / len; }
	fx[Y * DIM + X] += dx;
	fy[Y * DIM + X] += dy;
	rho[Y * DIM + X] = 10.0f;
	lmx = mx;
	lmy = my;
}

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{


 //    // uh...
 //    if (key == GLFW_KEY_ESCAPE) glfw_close(window);
 //    if (key == GLFW_KEY_M) glfwSetInputMode(window_manager->main_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); 
 //    if (key == GLFW_KEY_N) glfwSetInputMode(window_manager->main_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); 
 //    if (key == GLFW_KEY_V) glfwSwapInterval(1); // vsync
 //    if (key == GLFW_KEY_U) glfwSwapInterval(0); // unlocked

 //    // otherwise
 //    input.keyboard_state[key] = (action == GLFW_PRESS || action == GLFW_REPEAT) ? true : false; 

	if (key == GLFW_KEY_X) draw_smoke = 1 - draw_smoke;

	if (key == GLFW_KEY_A) frozen = 1 - frozen;

	if (key == GLFW_KEY_C) color_dir = 1 - color_dir; 

	if (key == GLFW_KEY_Y) draw_vecs = 1 - draw_vecs;

	if (key == GLFW_KEY_ESCAPE)  exit(0);



	// switch (key)
	// {
	//   case 't': dt -= 0.001; break;
	//   case 'T': dt += 0.001; break;
	//   case 'c': color_dir = 1 - color_dir; break;
	//   case 'S': vec_scale *= 1.2; break;
	//   case 's': vec_scale *= 0.8; break;
	//   case 'V': visc *= 5; break;
	//   case 'v': visc *= 0.2; break;
	//   case 'x': draw_smoke = 1 - draw_smoke;
	// 	    if (draw_smoke==0) draw_vecs = 1; break;
	//   case 'y': draw_vecs = 1 - draw_vecs;
	// 	    if (draw_vecs==0) draw_smoke = 1; break;
	//   case 'm': scalar_col++; if (scalar_col>COLOR_BANDS) scalar_col=COLOR_BLACKWHITE; break;
	//   case 'a': frozen = 1-frozen; break;
	//   case 'q': exit(0);
	// }
}


//main: The main program
int main(int argc, char **argv)
{
	printf("Fluid Flow Simulation and Visualization\n");
	printf("=======================================\n");
	printf("Click and drag the mouse to steer the flow!\n");
	printf("T/t:   increase/decrease simulation timestep\n");
	printf("S/s:   increase/decrease hedgehog scaling\n");
	printf("c:     toggle direction coloring on/off\n");
	printf("V/v:   increase decrease fluid viscosity\n");
	printf("x:     toggle drawing matter on/off\n");
	printf("y:     toggle drawing hedgehogs on/off\n");
	printf("m:     toggle thru scalar coloring\n");
	printf("a:     toggle the animation on/off\n");
	printf("q:     quit\n\n");

	// glutInit(&argc, argv);
	// glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	// glutInitWindowSize(500,500);
	// glutCreateWindow("Real-time smoke simulation and visualization");
	// glutDisplayFunc(display);
	// glutReshapeFunc(reshape);
	// glutIdleFunc(do_one_simulation_step);
	// glutKeyboardFunc(keyboard);
	// glutMotionFunc(drag);

	// init_simulation(DIM);	//initialize the simulation data structures
	// glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape


     // Request particular opengl version (4.5)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);

    //  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
    const int window_width = 500;
    const int window_height = 500;
    const char* title = "smoke sim";

    glfwInit();

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, title, nullptr, nullptr);


    winWidth = window_width;
    winHeight = window_height;


    assert(nullptr != window);

    glfwSetWindowPos(window, 20, 20);
    //@IMPORTANT!
    glfwMakeContextCurrent(window);

    //@IC(Sjors): gladLoadGL only after makeContextCurrent.    
    bool error = (gladLoadGL() == 0);
    if (error)
    {
        // logr::report_error("Failed to initialize OpenGL loader!\n");
        printf("failed to initialize openGL loader!\n");
        exit(1);
    }



    glfwSwapInterval(1); // Enable vsync
     // glfwSwapInterval(0); // explicitly disable vsync?

    // register callbacks
    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetCursorPosCallback(window, glfw_cursor_position_callback);
	glfwSetWindowSizeCallback(window, glfw_window_size_callback);

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
	init_simulation(DIM);	//initialize the simulation data 

	 /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

        /* Render here */
      	do_one_simulation_step();

      	display();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

    }


	// glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape

	return 0;
}