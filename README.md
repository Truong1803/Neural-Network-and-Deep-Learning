
Forward Propagation
### 1.	Cấu trúc mạng nơ-ron nhân tạo
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/2abl6qud09_image.png)
- Chú thích:
	- x1,x2… : là các giá trị đầu vào
	- y: là giá trị đầu ra
  	- w1,w2…: là các trọng số
  	- b1,b2… : là các bias
 
### 2.	Forward Propagation
Lan truyền xuôi là quá trình tính toán tính toán giá trị đầu ra để so sánh với giá trị thực tế từ các giá trị đầu vào thông qua phương trình zi = wi*xi + bi và hàm kích hoạt. Trong quá trình tính toán, các giá trị a sẽ được lưu lại tại mỗi tầng.
z gọi là giá trị pre-activation, còn a gọi là giá trị activation
- Tính toán lan truyền xuôi:
	- z1 = w1 * x1 + b1
	- a1 = act(z1)
	- z2 = w2 * a1 + b2
	- a2 = act(z2)
	- … 
	- ŷ = ai = act(zi)

![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/xz7vitzqa2_image.png)
Vậy tại sao cần các hàm kích hoạt
Giả sử a1=z1=w1*x1 + b1 , a2=z2 = w2*a1 + b2
a2 =  w2 *  ( w1 * x1 + b1 )  + b2 = w2*w1*a1 + w2*b1 + b2 
=>	a2 = w’*x + b’ .  Như vậy nếu không có hàm kích hoạt thì dù có bao nhiêu lớp ẩn thì thực tế sẽ chẳng có lớp ẩn nào cả. Khả năng dự đoán của mạng neural sẽ bị giới hạn và giảm đi rất nhiều.
### 3.	Một số hàm kích hoạt
#### 3.1	Hàm sigmoid
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/lfz3j2o9q8_image.png)
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/hw97j80vot_image.png)

Hàm Sigmoid nhận đầu vào là một số thực và chuyển thành một giá trị trong khoảng (0;1) (xem đồ thị phía trên). Đầu vào là số thực âm rất nhỏ sẽ cho đầu ra tiệm cận với 0, ngược lại, nếu đầu vào là một số thực dương lớn sẽ cho đầu ra là một số tiệm cận với 1.
Nhược điểm của hàm sigmoid: Một nhược điểm dễ nhận thấy là khi đầu vào có trị tuyệt đối lớn (rất âm hoặc rất dương), gradient của hàm số này sẽ rất gần với 0. Điều này đồng nghĩa với việc các hệ số tương ứng với unit đang xét sẽ gần như không được cập nhật.
#### 3.2 Hàm Tanh
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/fa6i3lonz7_image.png)
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/9pgbsir9q1_image.png)

Hàm Tanh nhận đầu vào là một số thực và chuyển thành một giá trị trong khoảng (-1; 1). Cũng như Sigmoid, hàm Tanh bị bão hoà ở 2 đầu (gradient thay đổi rất ít ở 2 đầu). Tuy nhiên hàm Tanh lại đối xứng qua 0 nên khắc phục được một nhược điểm của Sigmoid.

Thường được sử dụng trong các lớp ẩn của mạng nơ-ron vì giá trị của nó nằm trong khoảng từ -1 đến 1 do đó giá trị trung bình của lớp ẩn xuất hiện là 0 hoặc rất gần với nó, do đó giúp căn giữa dữ liệu bằng cách đưa giá trị trung bình gần bằng 0. Điều này làm cho học cho lớp tiếp theo dễ dàng hơn nhiều.
#### 3.3 Hàm ReLU
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/ngzako6fvq_image.png)
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/xn7vboecsa_image.png)

Hàm ReLU đang được sử dụng khá nhiều trong những năm gần đây khi huấn luyện các mạng neuron. ReLU đơn giản lọc các giá trị < 0.
- Ưu điểm: 
 	- Tốc độ hội tụ nhanh hơn hẳn. ReLU có tốc độ hội tụ nhanh gấp 6 lần Tanh. Điều này có thể do ReLU không bị bão hoà ở 2 đầu như Sigmoid và Tanh.
	- Tính toán nhanh hơn. Tanh và Sigmoid sử dụng hàm exp và công thức phức tạp hơn ReLU rất nhiều do vậy sẽ tốn nhiều chi phí hơn để tính toán. 
	- Hàm ReLU chỉ active các neural khi giá trị của neural >0. Vì vậy nó sẽ tạo ra tính thưa thớt (chỉ một số neural được active) cho các neural so với hàm sigmoid. Tính thưa thớt tạo ra khả năng phân loại tốt hơn cho neural network.
- Nhược điểm:
	- Với các node có giá trị nhỏ hơn 0, qua ReLU activation sẽ thành 0, hiện tượng đấy gọi là “Dying ReLU“. Nếu các node bị chuyển thành 0 thì sẽ không có ý nghĩa với bước linear activation ở lớp tiếp theo và các hệ số tương ứng từ node đấy cũng không được cập nhật với gradient descent.
	- Khi learning rate lớn, các trọng số (weights) có thể thay đổi theo cách làm tất cả neuron dừng việc cập nhật.

#### 3.4 Hàm Leaky ReLU
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/icooc0be7i_image.png)
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/x49w6ke6ey_image.png)

Leaky ReLU là một hàm khắc phục nhược điểm “dying ReLU” của hàm ReLu. Thay vì trả về giá trị 0 với các đầu vào <0 thì Leaky ReLU tạo ra một đường xiên có độ dốc nhỏ. Có nhiều báo cáo về việc hiệu Leaky ReLU có hiệu quả tốt hơn ReLU, nhưng hiệu quả này vẫn chưa rõ ràng và nhất quán.

### 4.	Vector hóa 
Nếu chúng ta có m mẫu cần huấn luyện thì sẽ cần đến một vòng lặp để tính toán các giá trị.
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/ri7gtjy4gs_image.png)

Nếu làm như vậy thì sẽ mất rất nhiều thời gian.
Vấn đề này sẽ được giải quyết bằng vector hóa.
Ta có ma trận X là các giá trị đầu vào được xếp thành các cột:

![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/d2kxrdscr4_image.png)

Tương tự như vậy với các giá trị còn lại ta sẽ có được các ma trận Zi,Ai,Wi,bi. Và công thức tính của chúng ta sẽ là:
![alt text](https://s3-ap-southeast-1.amazonaws.com/kipalog.com/tsxmon5m0v_image.png)

Làm như vậy ta sẽ loại bỏ được vòng lặp và làm thời gian tính toán trở nên hiệu quả hơn rất nhiều.

