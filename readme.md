# Segmenter-based-on-OpenMMLab
"Segmenter: Transformer for Semantic Segmentation" reproduced via mmsegmentation

We reproduce [Segmenter](https://arxiv.org/pdf/2105.05633.pdf) via [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 
based on [open-sourced code](https://github.com/rstrudel/segmenter).

## Results

### ADE20K

The passwds of download links are all 'nopw'
<table>
  <tr>
    <th>Exp</th>
    <th>Name</th>
    <th>backbone</th>
    <th>mIoU-SS </th>
    <th>Resolution</th>
    <th>BS</th>
    <th colspan="3">Download</th>
  </tr>
<tr>
    <td>4th line in Table3</td>
    <td>Seg-B<span>&#8224;</span>-Linear/16</td>
    <td>Deit</td>
    <td> 46.69 </td>
    <td>512x512</td>
    <td>8</td>
    <td><a href="https://pan.baidu.com/s/1M5oNIAjeiKjM22FydEUlMg">model</a></td>
    <td><a href="https://drive.google.com/file/d/17e-_fA8GnvZ-fJJVferKfzwG296lXYIp/view?usp=sharing">config</a></td>    
    <td><a href="https://drive.google.com/file/d/1wLjCIyG8gP5OUC0vImRVmOtxdm6kmZ-W/view?usp=sharing">log</a></td>
</tr>
<tr>
    <td>4th line in Table6</td>
    <td>Seg-B<span>&#8224;</span>-Mask/16</td>
    <td>Deit</td>
    <td> runing </td>
    <td>512x512</td>
    <td>8</td>
</tr>
</table>