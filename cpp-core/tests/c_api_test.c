#include "fpm/c_api.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"C ABI check failed at line %d\n",__LINE__); return 1; } } while(0)
int main(void)
{
	uint8_t pixels[64];
	for (int i=0;i<64;++i) pixels[i]=(uint8_t)(i*31+i*i);
	fpm_image image={pixels,8,8,8};
	fpm_pattern* pattern=NULL;
	char error[128];
	CHECK(fpm_abi_version()==1);
	CHECK(fpm_learn(image,256,&pattern,error,sizeof(error))==FPM_OK);
	fpm_config config=fpm_default_config(); config.max_targets=1; config.min_score=.99;
	size_t count=0;
	CHECK(fpm_match(image,pattern,&config,NULL,0,&count,error,sizeof(error))==FPM_BUFFER_TOO_SMALL);
	CHECK(count==1);
	fpm_result result;
	CHECK(fpm_match(image,pattern,&config,&result,1,&count,error,sizeof(error))==FPM_OK);
	CHECK(count==1 && result.score>.999 && result.center.x==4 && result.center.y==4);
	CHECK(fpm_match_pruned(image,pattern,&config,&result,1,&count,error,sizeof(error))==FPM_OK);
	CHECK(count==1 && result.score>.999 && result.center.x==4 && result.center.y==4);
	config.max_targets=0;
	CHECK(fpm_match(image,pattern,&config,&result,1,&count,error,sizeof(error))==FPM_ERROR);
	CHECK(count==0 && strlen(error)>0);
	fpm_destroy(pattern); fpm_destroy(NULL);
	pattern=NULL; image.stride=1;
	CHECK(fpm_learn(image,256,&pattern,error,sizeof(error))==FPM_ERROR && pattern==NULL);
	return 0;
}
