%Problem 2
data=csvread('/Users/rainsunny/Desktop/movies_csv/ratings.csv');

%Initialization
mean=[0 0 0 0 0];
covariance=0.1*eye(5);
t_v=zeros(5,5,942);
t_u=zeros(5,5,1682);
sum_expec_v=zeros(1,5,942);
sum_expec_u=zeros(1,5,1682);
u=mvnrnd(mean,covariance,943);
v=mvnrnd(mean,covariance,1682);

%read all columns from csv file
for i =1:size(data(:,1))
    read_data(data(i,1),data(i,2))=data(i,3);
end

%Set the iterations
%Update each u
    for i = 1:1682
        for k = 1:942
            if read_data(k,i)~=0
                t_u(:,:,i)=t_u(:,:,i)+u(k,:)'*u(k,:);
                sum_expec_u(:,:,i)=sum_expect_u(:,:,i)+eq(k,i)*u(k,:);
            end
        end
        t_u(:,:,i)=t_u(:,:,i)+eye(5);
        v(i,:)=(t_u(:,:,i)^(-1)*sum_expect_u(:,:,i)')';
    end
%Update each v
for q=1:100
    eq=abs(read_data).*(u*v')+read_data.*normpdf(-u*v')./normcdf(read_data.*(u*v'));
    for i = 1:942
        for k = 1:1682
            if read_r(i,k)~=0
                t_v(:,:,i)=t_v(:,:,i)+v(k,:)'*v(k,:);
                sum_expec_v(:,:,i)=sum_expec_v(:,:,i)+eq(i,k)*v(k,:);
            end
        end
        t_v(:,:,i)=t_v(:,:,i)+eye(5);
        u(i,:)=(t_v(:,:,i)^(-1)*sum_expec_v(:,:,i)')';
    end

    p(q)=(5/2)*log(1/(2*pi))*(942+1682)-(sum(diag(u*u'))+sum(diag(v*v')))/2+sum(sum(abs(read_data).*((0.5+0.5.*read_data).*log(normcdf(u*v'))+(0.5-0.5.*read_data).*log(normcdf(-u*v')))));
end

figure(1)

plot(20:100,p(20:100))
end










