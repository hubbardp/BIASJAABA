#include <stdarg.h>
#include <stdio.h>


#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(...) logger(1,__LINE__,__FUNCTION__,__FILE__,__VA_ARGS__);

static void logger(int is_error,const char *file,int line,const char* function,const char *fmt,...) {
    char buf1[1024]={0},buf2[1024]={0};
    va_list ap;
    va_start(ap,fmt);
    vsprintf_s(buf1,fmt,ap);
    va_end(ap);
#if 0
    sprintf_s(buf2,"%s(%d): %s()\n\t - %s\n",file,line,function,buf1);
#else
    sprintf_s(buf2,"%s\n",buf1);
#endif
    
}


