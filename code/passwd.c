#include <pwd.h>
#include <stddef.h>
#include <string.h>
#include <shadow.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char *argv[])
{
if(argc < 2)
{
printf("no usrname input");
return 1;
}
if (geteuid() != 0)
{
fprintf(stderr, "must be setuid root");
return -1;
} 
struct spwd *shd= getspnam(argv[1]);
if(shd != NULL)
{
static char crypt_char[80];
strcpy(crypt_char, shd->sp_pwdp);
char salt[13];
int i=0,j=0;
while(shd->sp_pwdp[i]!='\0')
{
salt[i]=shd->sp_pwdp[i];
if(salt[i]=='$')
{
j++;
if(j==3)
{
salt[i+1]='\0';
break;
}
}
i++;
}
if(j<3)
perror("file error or user cannot use.");
if(argc==3)
{
printf("salt: %s, crypt: %s\n", salt, crypt(argv[2], salt));
printf("shadowd passwd: %s\n", shd->sp_pwdp);
}
}
return 0;
}
